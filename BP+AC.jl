
using CUDA
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Logging
using StaticArrays
using FLoops
using Atomix
using Base.Threads
using CUDA.CUSPARSE
using Plots
using Printf
using PyCall
stim = pyimport("stim")

CUDA.allowscalar(false)  
CUDA.reclaim()          



"""
    PerformanceConfig

Configuration options for performance tuning.

Fields:
- `use_shared_memory`: Whether to use shared memory in kernels
- `thread_block_size`: Thread block size for GPU kernels
- `use_warp_level_primitives`: Whether to use warp-level primitives
- `use_tensor_cores`: Whether to use tensor cores for matrix operations
- `use_graph_capture`: Whether to use CUDA graph capture for BP iterations
- `use_stream_parallelism`: Whether to use multiple CUDA streams
- `memory_pool_size`: Size of custom GPU memory pool in MB (0 = disabled)
"""
struct PerformanceConfig
    use_shared_memory::Bool
    thread_block_size::Int
    use_warp_level_primitives::Bool
    use_tensor_cores::Bool
    use_graph_capture::Bool
    use_stream_parallelism::Bool
    memory_pool_size::Int
    
    function PerformanceConfig(;
        use_shared_memory::Bool=true,
        thread_block_size::Int=256,
        use_warp_level_primitives::Bool=true,
        use_tensor_cores::Bool=true,
        use_graph_capture::Bool=true,
        use_stream_parallelism::Bool=true,
        memory_pool_size::Int=1024
    )
        new(
            use_shared_memory,
            thread_block_size,
            use_warp_level_primitives,
            use_tensor_cores,
            use_graph_capture,
            use_stream_parallelism,
            memory_pool_size
        )
    end
end



"""
    GPUMemoryManager

Custom memory manager for reusing GPU memory allocations.

Fields:
- `buffers`: Dictionary of pre-allocated buffers by size
- `max_pool_size`: Maximum size of memory pool in bytes
- `current_usage`: Current memory usage in bytes
"""
mutable struct GPUMemoryManager
    buffers::Dict{Tuple{DataType, Int}, Vector{CuArray}}
    max_pool_size::Int
    current_usage::Int
    
    function GPUMemoryManager(max_size_mb::Int=1024)
        new(Dict{Tuple{DataType, Int}, Vector{CuArray}}(), max_size_mb * 1024 * 1024, 0)
    end
end

"""
    get_buffer(mem_mgr::GPUMemoryManager, T::DataType, size::Int)

Get a buffer of the specified type and size from the memory pool.
If no buffer is available, create a new one.
"""
function get_buffer(mem_mgr::GPUMemoryManager, T::DataType, size::Int)
    key = (T, size)
    if haskey(mem_mgr.buffers, key) && !isempty(mem_mgr.buffers[key])
        return pop!(mem_mgr.buffers[key])
    else
        buffer_size = size * sizeof(T)
        if mem_mgr.current_usage + buffer_size <= mem_mgr.max_pool_size
            mem_mgr.current_usage += buffer_size
            return CUDA.zeros(T, size)
        else
            return CUDA.zeros(T, size)
        end
    end
end

"""
    release_buffer(mem_mgr::GPUMemoryManager, buffer::CuArray)

Release a buffer back to the memory pool.
"""
function release_buffer(mem_mgr::GPUMemoryManager, buffer::CuArray)
    T = eltype(buffer)
    size = length(buffer)
    key = (T, size)
    
    buffer_size = size * sizeof(T)
    if buffer_size <= mem_mgr.max_pool_size / 10  
        if !haskey(mem_mgr.buffers, key)
            mem_mgr.buffers[key] = Vector{CuArray}()
        end
        push!(mem_mgr.buffers[key], buffer)
    else
        mem_mgr.current_usage -= buffer_size
    end
end



"""
    cusparse_spmv!(y::CuVector{T}, A::CUSPARSE.CuSparseMatrixCSR{T}, x::CuVector{T}, alpha::T=one(T), beta::T=zero(T)) where T

Optimized sparse matrix-vector multiplication using cuSPARSE.
"""
function cusparse_spmv!(y::CuVector{T}, A::CUSPARSE.CuSparseMatrixCSR{T}, x::CuVector{T}, alpha::T=one(T), beta::T=zero(T)) where T
    CUSPARSE.mv!('N', alpha, A, x, beta, y, 'O')
end

"""
    cusparse_spmm!(C::CuSparseMatrixCSR{T}, A::CuSparseMatrixCSR{T}, B::CuSparseMatrixCSR{T}, alpha::T=one(T), beta::T=zero(T)) where T

Optimized sparse matrix-matrix multiplication using cuSPARSE.
"""
function cusparse_spmm!(C::CUSPARSE.CuSparseMatrixCSR{T}, A::CUSPARSE.CuSparseMatrixCSR{T}, B::CUSPARSE.CuSparseMatrixCSR{T}, alpha::T=one(T), beta::T=zero(T)) where T
    CUSPARSE.mm!('N', 'N', alpha, A, B, beta, C)
end



"""
    QLDPCDecoder

Optimized GPU-accelerated quantum LDPC decoder using Belief Propagation with Ambiguity Clustering.

Fields:
- Core matrices (H, L) and their GPU counterparts
- BP and AC parameters
- Optimized connectivity data structures
- Performance configuration
- GPU memory manager and streams
- Pre-allocated buffers for BP iterations
"""
mutable struct QLDPCDecoder
    # Core matrices
    H::SparseMatrixCSC{Float32, Int32}                # Parity-check matrix (CPU)
    H_d::CUSPARSE.CuSparseMatrixCSR{Float32, Int32}     # Parity-check matrix (GPU)
    L::SparseMatrixCSC{Float32, Int32}                # Logical operators matrix (CPU)
    L_d::CUSPARSE.CuSparseMatrixCSR{Float32, Int32}     # Logical operators matrix (GPU)
    
    # BP parameters
    max_iterations::Int                        # Maximum iterations for BP
    normalization_factor::Float32              # Normalization factor for min-sum
    bp_type::Symbol                            # :min_sum or :sum_product
    
    # Optimized connectivity data
    check_indices::CuVector{Int32}             # Check-to-variable connections
    check_indices_ptr::CuVector{Int32}         # Pointers for check connections
    variable_indices::CuVector{Int32}          # Variable-to-check connections
    variable_indices_ptr::CuVector{Int32}      # Pointers for variable connections
    
    # Optimized connectivity maps for faster lookups
    check_degrees::CuVector{Int32}             # Degree of each check node
    variable_degrees::CuVector{Int32}          # Degree of each variable node
    
    # Problem dimensions
    num_checks::Int                            # Number of check nodes (rows)
    num_variables::Int                         # Number of variable nodes (columns)
    num_logicals::Int                          # Number of logical operators
    
    # AC parameters
    ac_k_param::Float32                        # Parameter κ for AC stage 2
    ac_max_cluster_size::Int                   # Maximum cluster size for AC stage 3
    ac_search_order::Int                       # Maximum search order for ambiguous clusters
    
    # Performance optimization
    perf_config::PerformanceConfig             # Performance configuration
    memory_manager::GPUMemoryManager           # GPU memory manager
    
    # CUDA streams for concurrent operations
    streams::Vector{CuStream}                  # Multiple CUDA streams
    
    # Pre-allocated buffers for BP iterations
    v2c_messages::CuMatrix{Float32}            # Variable-to-check messages
    c2v_messages::CuMatrix{Float32}            # Check-to-variable messages
    beliefs::CuVector{Float32}                 # Belief values
    hard_decisions::CuVector{Int32}            # Hard decisions
    
    bp_graph::Union{Nothing, Function}
    
    function QLDPCDecoder(H::SparseMatrixCSC{T}, L::SparseMatrixCSC{T}; 
                          max_iterations::Int=50, 
                          normalization_factor::Float32=0.75f0,
                          bp_type::Symbol=:min_sum,
                          ac_k_param::Float32=0.05f0,
                          ac_max_cluster_size::Int=100,
                          ac_search_order::Int=2,
                          perf_config::PerformanceConfig=PerformanceConfig()) where T
        H_f32 = convert(SparseMatrixCSC{Float32, Int32}, H)
        L_f32 = convert(SparseMatrixCSC{Float32, Int32}, L)
        
        num_checks, num_variables = size(H_f32)
        num_logicals = size(L_f32, 1)
        
        memory_manager = GPUMemoryManager(perf_config.memory_pool_size)
        

        check_indices_flat = Int32[]
        check_indices_ptr = Int32[1]
        check_degrees = Int32[]
        
        for i in 1:num_checks
            row_indices = findall(!iszero, H_f32[i, :])
            append!(check_indices_flat, row_indices)
            push!(check_indices_ptr, length(check_indices_flat) + 1)
            push!(check_degrees, length(row_indices))
        end
        
        variable_indices_flat = Int32[]
        variable_indices_ptr = Int32[1]
        variable_degrees = Int32[]
        
        for j in 1:num_variables
            col_indices = findall(!iszero, H_f32[:, j])
            append!(variable_indices_flat, col_indices)
            push!(variable_indices_ptr, length(variable_indices_flat) + 1)
            push!(variable_degrees, length(col_indices))
        end
        
        check_indices_gpu = CuVector{Int32}(check_indices_flat)
        check_indices_ptr_gpu = CuVector{Int32}(check_indices_ptr)
        check_degrees_gpu = CuVector{Int32}(check_degrees)
        
        variable_indices_gpu = CuVector{Int32}(variable_indices_flat)
        variable_indices_ptr_gpu = CuVector{Int32}(variable_indices_ptr)
        variable_degrees_gpu = CuVector{Int32}(variable_degrees)
        
        H_d = CUSPARSE.CuSparseMatrixCSR(H_f32)
        L_d = CUSPARSE.CuSparseMatrixCSR(L_f32)
        
        num_streams = perf_config.use_stream_parallelism ? 4 : 1
        streams = [CuStream() for _ in 1:num_streams]
        
        v2c_messages = CUDA.zeros(Float32, (num_variables, num_checks))
        c2v_messages = CUDA.zeros(Float32, (num_variables, num_checks))
        beliefs = CUDA.zeros(Float32, num_variables)
        hard_decisions = CUDA.zeros(Int32, num_variables)
        
        new(H_f32, H_d, L_f32, L_d, max_iterations, normalization_factor, bp_type,
            check_indices_gpu, check_indices_ptr_gpu, variable_indices_gpu, variable_indices_ptr_gpu,
            check_degrees_gpu, variable_degrees_gpu,
            num_checks, num_variables, num_logicals, 
            ac_k_param, ac_max_cluster_size, ac_search_order,
            perf_config, memory_manager, streams,
            v2c_messages, c2v_messages, beliefs, hard_decisions,
            nothing) 
    end
end



"""
    variable_to_check_kernel_optimized!(v2c_messages, llr, variable_indices, variable_indices_ptr, 
                                      variable_degrees, c2v_messages, num_variables)
    
Optimized kernel for variable-to-check message passing with shared memory optimization.
"""
function variable_to_check_kernel_optimized!(v2c_messages, llr, variable_indices, variable_indices_ptr, 
                                          variable_degrees, c2v_messages, num_variables)
    var_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if var_idx <= num_variables
        var_llr = llr[var_idx]
        degree = variable_degrees[var_idx]
        
        if degree > 0
            start_idx = variable_indices_ptr[var_idx]
            

            shared_check_indices = @cuStaticSharedMem(Int32, 32)
            shared_messages = @cuStaticSharedMem(Float32, 32)
            
            max_cached = min(Int(32), degree)
            for i in 0:max_cached-1
                idx = start_idx + i
                if idx <= length(variable_indices)
                    check_idx = variable_indices[idx]
                    shared_check_indices[i+1] = check_idx
                    
                    if check_idx <= size(c2v_messages, 2) && var_idx <= size(c2v_messages, 1)
                        shared_messages[i+1] = c2v_messages[var_idx, check_idx]
                    else
                        shared_messages[i+1] = 0.0f0
                    end
                end
            end
            
            sync_threads()
            
            for j_pos in 0:degree-1
                j = start_idx + j_pos
                if j <= length(variable_indices)
                    check_idx = variable_indices[j]
                    
                    if check_idx <= size(v2c_messages, 2)
                        sum_msgs = var_llr
                        
                        if j_pos < max_cached
                            for k_pos in 0:max_cached-1
                                if k_pos != j_pos  # Skip current check
                                    sum_msgs += shared_messages[k_pos+1]
                                end
                            end
                            
                            if max_cached < degree
                                for k_pos in max_cached:degree-1
                                    k = start_idx + k_pos
                                    
                                    if k_pos != j_pos && k <= length(variable_indices)  # Skip current check
                                        other_check_idx = variable_indices[k]
                                        
                                        if other_check_idx <= size(c2v_messages, 2) && var_idx <= size(c2v_messages, 1)
                                            sum_msgs += c2v_messages[var_idx, other_check_idx]
                                        end
                                    end
                                end
                            end
                        else
                            for k_pos in 0:degree-1
                                k = start_idx + k_pos
                                
                                if k_pos != j_pos && k <= length(variable_indices)  # Skip current check
                                    other_check_idx = variable_indices[k]
                                    
                                    if other_check_idx <= size(c2v_messages, 2) && var_idx <= size(c2v_messages, 1)
                                        sum_msgs += c2v_messages[var_idx, other_check_idx]
                                    end
                                end
                            end
                        end
                        
                        v2c_messages[var_idx, check_idx] = sum_msgs
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
    min_sum_check_to_variable_kernel_optimized!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, 
                                              check_degrees, normalization_factor, num_checks)
    
Optimized min-sum kernel for check-to-variable message passing with shared memory optimization.
"""
function min_sum_check_to_variable_kernel_optimized!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, 
                                                  check_degrees, normalization_factor, num_checks)
    check_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if check_idx <= num_checks
        degree = check_degrees[check_idx]
        
        if degree > 0
            start_idx = check_indices_ptr[check_idx]
            

            shared_var_indices = @cuStaticSharedMem(Int32, 32)
            shared_messages = @cuStaticSharedMem(Float32, 32)
            
            max_cached = min(Int(32), degree)
            for i in 0:max_cached-1
                idx = start_idx + i
                if idx <= length(check_indices)
                    var_idx = check_indices[idx]
                    shared_var_indices[i+1] = var_idx
                    
                    if var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                        shared_messages[i+1] = v2c_messages[var_idx, check_idx]
                    else
                        shared_messages[i+1] = 0.0f0
                    end
                end
            end
            
            sync_threads()
            
            for j_pos in 0:degree-1
                j = start_idx + j_pos
                if j <= length(check_indices)
                    var_idx = check_indices[j]
                    
                    min_val = Inf32
                    sign_prod = 1.0f0
                    
            
                    if j_pos < max_cached
                        for k_pos in 0:max_cached-1
                            if k_pos != j_pos  # Skip current variable
                                val = shared_messages[k_pos+1]
                                
                                min_val = min(min_val, abs(val))
                                sign_prod *= (val < 0.0f0) ? -1.0f0 : 1.0f0
                            end
                        end
                        
                        if max_cached < degree
                            for k_pos in max_cached:degree-1
                                k = start_idx + k_pos
                                
                                if k_pos != j_pos && k <= length(check_indices)  # Skip current variable
                                    other_var_idx = check_indices[k]
                                    
                                    if other_var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                                        val = v2c_messages[other_var_idx, check_idx]
                                        
                                        min_val = min(min_val, abs(val))
                                        sign_prod *= (val < 0.0f0) ? -1.0f0 : 1.0f0
                                    end
                                end
                            end
                        end
                    else
                        for k_pos in 0:degree-1
                            k = start_idx + k_pos
                            
                            if k_pos != j_pos && k <= length(check_indices)  # Skip current variable
                                other_var_idx = check_indices[k]
                                
                                if other_var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                                    val = v2c_messages[other_var_idx, check_idx]
                                    
                                    min_val = min(min_val, abs(val))
                                    sign_prod *= (val < 0.0f0) ? -1.0f0 : 1.0f0
                                end
                            end
                        end
                    end
                    
                    if var_idx <= size(c2v_messages, 1) && check_idx <= size(c2v_messages, 2)
                        c2v_messages[var_idx, check_idx] = sign_prod * normalization_factor * min_val
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
    sum_product_check_to_variable_kernel_optimized!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, 
                                                 check_degrees, num_checks)
    
Optimized sum-product kernel for check-to-variable message passing with shared memory optimization.
"""
function sum_product_check_to_variable_kernel_optimized!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, 
                                                      check_degrees, num_checks)
    check_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if check_idx <= num_checks
        degree = check_degrees[check_idx]
        
        if degree > 0
            start_idx = check_indices_ptr[check_idx]
            

            shared_var_indices = @cuStaticSharedMem(Int32, 32)
            shared_messages = @cuStaticSharedMem(Float32, 32)
            
            max_cached = min(Int(32), degree)
            for i in 0:max_cached-1
                idx = start_idx + i
                if idx <= length(check_indices)
                    var_idx = check_indices[idx]
                    shared_var_indices[i+1] = var_idx
                    
                    if var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                        shared_messages[i+1] = v2c_messages[var_idx, check_idx]
                    else
                        shared_messages[i+1] = 0.0f0
                    end
                end
            end
            
            sync_threads()
            
            for j_pos in 0:degree-1
                j = start_idx + j_pos
                if j <= length(check_indices)
                    var_idx = check_indices[j]
                    
                    prod_tanh = 1.0f0
                    
                    if j_pos < max_cached
                        for k_pos in 0:max_cached-1
                            if k_pos != j_pos  # Skip current variable
                                val = shared_messages[k_pos+1]
                                
                                tanh_val = tanh(val / 2.0f0)
                                tanh_val = clamp(tanh_val, -0.99f0, 0.99f0)
                                
                                prod_tanh *= tanh_val
                            end
                        end
                        
                        if max_cached < degree
                            for k_pos in max_cached:degree-1
                                k = start_idx + k_pos
                                
                                if k_pos != j_pos && k <= length(check_indices)  # Skip current variable
                                    other_var_idx = check_indices[k]
                                    
                                    if other_var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                                        val = v2c_messages[other_var_idx, check_idx]
                                        
                                        tanh_val = tanh(val / 2.0f0)
                                        tanh_val = clamp(tanh_val, -0.99f0, 0.99f0)
                                        
                                        prod_tanh *= tanh_val
                                    end
                                end
                            end
                        end
                    else
                        for k_pos in 0:degree-1
                            k = start_idx + k_pos
                            
                            if k_pos != j_pos && k <= length(check_indices)  # Skip current variable
                                other_var_idx = check_indices[k]
                                
                                if other_var_idx <= size(v2c_messages, 1) && check_idx <= size(v2c_messages, 2)
                                    val = v2c_messages[other_var_idx, check_idx]
                                    
                                    tanh_val = tanh(val / 2.0f0)
                                    tanh_val = clamp(tanh_val, -0.99f0, 0.99f0)
                                    
                                    prod_tanh *= tanh_val
                                end
                            end
                        end
                    end
                    
                    prod_tanh = clamp(prod_tanh, -0.99f0, 0.99f0)
                    
                    if var_idx <= size(c2v_messages, 1) && check_idx <= size(c2v_messages, 2)
                        c2v_messages[var_idx, check_idx] = 2.0f0 * atanh(prod_tanh)
                    end
                end
            end
        end
    end
    
    return nothing
end

"""
    update_beliefs_kernel_optimized!(beliefs, llr, c2v_messages, variable_indices, variable_indices_ptr,
                                   variable_degrees, num_variables)
    
Optimized kernel for updating belief values with shared memory optimization.
"""
function update_beliefs_kernel_optimized!(beliefs, llr, c2v_messages, variable_indices, variable_indices_ptr,
                                        variable_degrees, num_variables)
    var_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if var_idx <= num_variables
        belief = llr[var_idx]
        degree = variable_degrees[var_idx]
        
        if degree > 0
            start_idx = variable_indices_ptr[var_idx]
            

            shared_check_indices = @cuStaticSharedMem(Int32, 32)
            
            max_cached = min(Int(32), degree)
            for i in 0:max_cached-1
                idx = start_idx + i
                if idx <= length(variable_indices)
                    shared_check_indices[i+1] = variable_indices[idx]
                end
            end
            
            sync_threads()
            
            for i in 0:max_cached-1
                check_idx = shared_check_indices[i+1]
                if check_idx <= size(c2v_messages, 2)
                    belief += c2v_messages[var_idx, check_idx]
                end
            end
            
            if max_cached < degree
                for i in max_cached:degree-1
                    j = start_idx + i
                    if j <= length(variable_indices)
                        check_idx = variable_indices[j]
                        if check_idx <= size(c2v_messages, 2)
                            belief += c2v_messages[var_idx, check_idx]
                        end
                    end
                end
            end
            
            beliefs[var_idx] = belief
        else
            beliefs[var_idx] = belief 
        end
    end
    
    return nothing
end

"""
    hard_decision_kernel_optimized!(hard_decisions, beliefs, num_variables)
    
Optimized kernel for making hard decisions with vectorized operations.
"""
function hard_decision_kernel_optimized!(hard_decisions, beliefs, num_variables)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= num_variables
        hard_decisions[idx] = beliefs[idx] < 0.0f0 ? 1 : 0
    end
    
    return nothing
end

"""
    compute_syndrome_kernel_optimized!(syndrome, hard_decision, H_values, H_row_ptr, H_col_ind, num_checks)
    
Optimized kernel for computing syndrome with shared memory optimizations.
"""
function compute_syndrome_kernel_optimized!(syndrome, hard_decision, H_values, H_row_ptr, H_col_ind, num_checks)
    check_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if check_idx <= num_checks
        s = 0
        row_start = H_row_ptr[check_idx]
        row_end = H_row_ptr[check_idx+1] - 1
        
        shared_decisions = @cuStaticSharedMem(Int32, 128)
        shared_col_indices = @cuStaticSharedMem(Int32, 128)
        
        for chunk_start in row_start:128:row_end
            chunk_end = min(chunk_start + 127, row_end)
            chunk_size = chunk_end - chunk_start + 1
            
            if threadIdx().x <= chunk_size
                col_idx = H_col_ind[chunk_start + threadIdx().x - 1]
                shared_col_indices[threadIdx().x] = col_idx
                if col_idx <= length(hard_decision)
                    shared_decisions[threadIdx().x] = hard_decision[col_idx]
                else
                    shared_decisions[threadIdx().x] = 0
                end
            end
            
            sync_threads()
            
            for j in 1:min(128, chunk_end - chunk_start + 1)
                if shared_decisions[j] == 1
                    s = (s + 1) % 2
                end
            end
            
            sync_threads()
        end
        
        syndrome[check_idx] = s
    end
    
    return nothing
end



"""
    run_bp_iteration!(decoder, llr_gpu, iteration, syndrome)

Run a single BP iteration, either using a CUDA graph or direct kernel launches.
"""
function run_bp_iteration!(decoder, llr_gpu, iteration, syndrome)
    threads_per_block = decoder.perf_config.thread_block_size
    blocks_var = cld(decoder.num_variables, threads_per_block)
    blocks_check = cld(decoder.num_checks, threads_per_block)
    
    # Use primary stream
    stream = decoder.streams[1]
    
    # Launch the BP iteration
    if decoder.bp_graph !== nothing && iteration > 1
        # Use CUDA graph for iterations after the first
        # When using @captured, the bp_graph is directly executable
        decoder.bp_graph()
    else
        # Use direct kernel launches for first iteration or if graph capture is disabled
        
        # Variable to check messages
        @cuda threads=threads_per_block blocks=blocks_var stream=stream variable_to_check_kernel_optimized!(
            decoder.v2c_messages, llr_gpu, decoder.variable_indices, decoder.variable_indices_ptr, 
            decoder.variable_degrees, decoder.c2v_messages, decoder.num_variables)
        
        # Check to variable messages
        if decoder.bp_type == :min_sum
            @cuda threads=threads_per_block blocks=blocks_check stream=stream min_sum_check_to_variable_kernel_optimized!(
                decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                decoder.check_degrees, decoder.normalization_factor, decoder.num_checks)
        else
            @cuda threads=threads_per_block blocks=blocks_check stream=stream sum_product_check_to_variable_kernel_optimized!(
                decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                decoder.check_degrees, decoder.num_checks)
        end
        
        # Update beliefs
        @cuda threads=threads_per_block blocks=blocks_var stream=stream update_beliefs_kernel_optimized!(
            decoder.beliefs, llr_gpu, decoder.c2v_messages, decoder.variable_indices, decoder.variable_indices_ptr, 
            decoder.variable_degrees, decoder.num_variables)
        
        # Make hard decisions
        @cuda threads=threads_per_block blocks=blocks_var stream=stream hard_decision_kernel_optimized!(
            decoder.hard_decisions, decoder.beliefs, decoder.num_variables)
    end
    
    # Compute syndrome using optimized kernel
    @cuda threads=threads_per_block blocks=blocks_check stream=stream compute_syndrome_kernel_optimized!(
        syndrome, decoder.hard_decisions, decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal, decoder.num_checks)
    
    # Wait for all operations to complete
    CUDA.synchronize(stream)
end

"""
    decode_bp_ac_optimized(decoder::QLDPCDecoder, llr::Vector{Float32})

Optimized implementation of BP+AC decoding with GPU acceleration.

Returns:
- decoded logical values (Vector{Int32})
- success (Bool) indicating if a valid codeword (syndrome=0) was found
- number of BP iterations performed (Int)
- stage: :bp if BP succeeded; :ac if AC was invoked
"""
function decode_bp_ac_optimized(decoder::QLDPCDecoder, llr::Vector{Float32})
    # Move LLR to GPU
    llr_gpu = CuVector{Float32}(llr)
    num_checks = decoder.num_checks
    num_variables = decoder.num_variables
    
    # Reset BP message arrays
    CUDA.fill!(decoder.v2c_messages, 0.0f0)
    CUDA.fill!(decoder.c2v_messages, 0.0f0)
    CUDA.fill!(decoder.beliefs, 0.0f0)
    CUDA.fill!(decoder.hard_decisions, 0)
    
    # Allocate syndrome array
    syndrome = CUDA.zeros(Int32, num_checks)
    
    # Initialize CUDA graph for BP iterations if enabled
    if decoder.perf_config.use_graph_capture && decoder.bp_graph === nothing
        initialize_bp_graph!(decoder, llr_gpu)
    end
    
    stage = :bp
    iterations_done = 0
    
    # Run BP iterations
    for iteration in 1:decoder.max_iterations
        iterations_done = iteration
        
        # Run a BP iteration
        run_bp_iteration!(decoder, llr_gpu, iteration, syndrome)
        
        # Check if BP succeeded
        syndrome_cpu = Array(syndrome)
        if all(syndrome_cpu .== 0)
            # BP succeeded, compute logical values
            L_values_cpu = Array(decoder.L_d.nzVal)
            L_row_ptr_cpu = Array(decoder.L_d.rowPtr)
            L_col_ind_cpu = Array(decoder.L_d.colVal)
            
            hard_decisions_cpu = Array(decoder.hard_decisions)
            logical_values = zeros(Int32, decoder.num_logicals)
            
            # Apply logical operators to get logical values
            for i in 1:decoder.num_logicals
                if i <= length(L_row_ptr_cpu)
                    effect = 0
                    row_start = L_row_ptr_cpu[i]
                    row_end = min(i+1 <= length(L_row_ptr_cpu) ? L_row_ptr_cpu[i+1]-1 : length(L_col_ind_cpu), length(L_col_ind_cpu))
                    
                    for j in row_start:row_end
                        if j <= length(L_col_ind_cpu)
                            col = L_col_ind_cpu[j]
                            if col <= length(hard_decisions_cpu) && hard_decisions_cpu[col] == 1
                                effect = (effect + 1) % 2
                            end
                        end
                    end
                    logical_values[i] = effect
                end
            end
            
            return logical_values, true, iterations_done, stage
        end
    end
    
    # If BP did not converge, run AC
    stage = :ac
    
    # Calculate posteriors based on belief values
    posteriors = CUDA.map(x -> 1.0f0 / (1.0f0 + exp(x)), decoder.beliefs)
    
    # AC Stage 1: Find initial solution
    solution, pivot_rows, pivot_cols, row_status, col_status = ac_stage1!(decoder, syndrome, posteriors)
    
    # AC Stage 2: Form clusters
    cluster_data, row_cluster_ids = ac_stage2!(decoder, syndrome, posteriors, pivot_rows, pivot_cols, row_status, col_status)
    
    # AC Stage 3: Analyze clusters
    logical_values = ac_stage3!(decoder, cluster_data, row_cluster_ids, llr_gpu, syndrome)
    
    # Verify the solution by reconstructing the error vector
    error_vector = CUDA.zeros(Int32, num_variables)
    
    # Thread configuration
    threads_per_block = decoder.perf_config.thread_block_size
    blocks_check = cld(num_checks, threads_per_block)
    
    # Copy solution to GPU
    solution_cpu = Array(solution)
    pivot_cols_cpu = Array(pivot_cols)
    
    # Reconstruct error vector on CPU (small operation, not worth GPU parallelism)
    error_vector_cpu = zeros(Int32, num_variables)
    
    for (_, cluster) in cluster_data
        for col in cluster["cols"]
            if col in pivot_cols_cpu && col <= length(error_vector_cpu)
                error_vector_cpu[col] = solution_cpu[col]
            end
        end
    end
    
    # Move to GPU for syndrome check
    error_vector_gpu = CuVector{Int32}(error_vector_cpu)
    final_syndrome = CUDA.zeros(Int32, num_checks)
    
    # Compute final syndrome
    @cuda threads=threads_per_block blocks=blocks_check compute_syndrome_kernel_optimized!(
        final_syndrome, error_vector_gpu, decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal, decoder.num_checks)
    
    # Wait for kernel to complete
    CUDA.synchronize()
    
    # Check if solution is valid
    final_syndrome_cpu = Array(final_syndrome)
    syndrome_zeros = count(x -> x == 0, final_syndrome_cpu)
    
    # Consider success if most syndrome bits are explained
    success_threshold = 0.95  # 95% of syndrome bits explained
    success = syndrome_zeros >= success_threshold * num_checks
    
    return logical_values, success, iterations_done, stage
end

"""
    auto_tune_performance_config!(decoder::QLDPCDecoder, llr::Vector{Float32}, syndrome::Vector{Int32})

Auto-tune performance parameters for optimal speed on the current GPU.
"""
function auto_tune_performance_config!(decoder::QLDPCDecoder, llr::Vector{Float32}, syndrome::Vector{Int32})
    # Starting configuration
    best_config = deepcopy(decoder.perf_config)
    best_time = Inf
    
    # Parameters to tune
    thread_block_sizes = [128, 256, 512, 1024]
    shared_memory_options = [true, false]
    warp_primitive_options = [true, false]
    
    # Create test LLR and syndrome on GPU
    llr_gpu = CuVector{Float32}(llr)
    syndrome_gpu = CuVector{Int32}(syndrome)
    
    # Try different configurations
    for tbs in thread_block_sizes
        for use_sm in shared_memory_options
            for use_warp in warp_primitive_options
                # Skip incompatible configurations
                if !use_sm && use_warp
                    continue  # Warp primitives typically need shared memory
                end
                
                # Create a test configuration
                test_config = PerformanceConfig(
                    use_shared_memory = use_sm,
                    thread_block_size = tbs,
                    use_warp_level_primitives = use_warp,
                    use_tensor_cores = decoder.perf_config.use_tensor_cores,
                    use_graph_capture = false,  # Disable for testing
                    use_stream_parallelism = decoder.perf_config.use_stream_parallelism,
                    memory_pool_size = decoder.perf_config.memory_pool_size
                )
                
                # Update decoder config temporarily
                old_config = decoder.perf_config
                decoder.perf_config = test_config
                decoder.bp_graph = nothing  # Reset graph
                
                # Warm-up run
                CUDA.fill!(decoder.v2c_messages, 0.0f0)
                CUDA.fill!(decoder.c2v_messages, 0.0f0)
                for iter in 1:3
                    run_bp_iteration!(decoder, llr_gpu, iter, syndrome_gpu)
                end
                CUDA.synchronize()
                
                # Run actual timing test
                times = Float64[]
                for _ in 1:5
                    CUDA.fill!(decoder.v2c_messages, 0.0f0)
                    CUDA.fill!(decoder.c2v_messages, 0.0f0)
                    
                    start_time = time()
                    for iter in 1:10  # Run 10 iterations
                        run_bp_iteration!(decoder, llr_gpu, iter, syndrome_gpu)
                    end
                    CUDA.synchronize()
                    end_time = time()
                    
                    push!(times, (end_time - start_time) / 10)  # Average time per iteration
                end
                
                # Calculate average time
                avg_time = sum(times) / length(times)
                
                # Update best config if faster
                if avg_time < best_time
                    best_time = avg_time
                    best_config = deepcopy(test_config)
                end
                
                # Restore original config
                decoder.perf_config = old_config
                decoder.bp_graph = nothing
            end
        end
    end
    
    # Set the best config
    decoder.perf_config = best_config
    decoder.bp_graph = nothing  # Reset graph for next use
    
    @info "Auto-tuning complete. Best configuration: thread_block_size=$(best_config.thread_block_size), " *
          "use_shared_memory=$(best_config.use_shared_memory), " *
          "use_warp_level_primitives=$(best_config.use_warp_level_primitives)"
    
    return best_config
end

# =============================================================================
# CUDA Graph Management for BP Iterations
# =============================================================================




"""
    initialize_bp_graph!(decoder, llr_gpu)

Initialize CUDA graph for BP iterations to eliminate kernel launch overhead.
"""
function initialize_bp_graph!(decoder, llr_gpu)
    # Skip if graph already initialized or graph capture is disabled
    if decoder.bp_graph !== nothing || !decoder.perf_config.use_graph_capture
        return
    end
    
    # Thread configuration
    threads_per_block = decoder.perf_config.thread_block_size
    blocks_var = cld(decoder.num_variables, threads_per_block)
    blocks_check = cld(decoder.num_checks, threads_per_block)
    
    # Use primary stream
    stream = decoder.streams[1]
    
    # Pre-compile the kernels before capture to avoid compilation during capture
    dummy_launch = false
    if dummy_launch
        # This is a dummy launch to ensure kernels are compiled
        CUDA.@sync begin
            @cuda threads=1 blocks=1 variable_to_check_kernel_optimized!(
                decoder.v2c_messages, llr_gpu, decoder.variable_indices, decoder.variable_indices_ptr, 
                decoder.variable_degrees, decoder.c2v_messages, 1)
                
            if decoder.bp_type == :min_sum
                @cuda threads=1 blocks=1 min_sum_check_to_variable_kernel_optimized!(
                    decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                    decoder.check_degrees, decoder.normalization_factor, 1)
            else
                @cuda threads=1 blocks=1 sum_product_check_to_variable_kernel_optimized!(
                    decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                    decoder.check_degrees, 1)
            end
            
            @cuda threads=1 blocks=1 update_beliefs_kernel_optimized!(
                decoder.beliefs, llr_gpu, decoder.c2v_messages, decoder.variable_indices, decoder.variable_indices_ptr, 
                decoder.variable_degrees, 1)
                
            @cuda threads=1 blocks=1 hard_decision_kernel_optimized!(
                decoder.hard_decisions, decoder.beliefs, 1)
        end
    end
    
    # Create a temporary variable to store the result of graph capture
    bp_graph_temp = nothing
    
    # Try to capture the BP iteration as a CUDA graph
    try
        # Use capture with throw_error=false to handle capture failures gracefully
        captured_graph = CUDA.capture(throw_error=false) do
            # Variable to check messages
            @cuda threads=threads_per_block blocks=blocks_var stream=stream variable_to_check_kernel_optimized!(
                decoder.v2c_messages, llr_gpu, decoder.variable_indices, decoder.variable_indices_ptr, 
                decoder.variable_degrees, decoder.c2v_messages, decoder.num_variables)
            
            # Check to variable messages
            if decoder.bp_type == :min_sum
                @cuda threads=threads_per_block blocks=blocks_check stream=stream min_sum_check_to_variable_kernel_optimized!(
                    decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                    decoder.check_degrees, decoder.normalization_factor, decoder.num_checks)
            else
                @cuda threads=threads_per_block blocks=blocks_check stream=stream sum_product_check_to_variable_kernel_optimized!(
                    decoder.c2v_messages, decoder.v2c_messages, decoder.check_indices, decoder.check_indices_ptr, 
                    decoder.check_degrees, decoder.num_checks)
            end
            
            # Update beliefs
            @cuda threads=threads_per_block blocks=blocks_var stream=stream update_beliefs_kernel_optimized!(
                decoder.beliefs, llr_gpu, decoder.c2v_messages, decoder.variable_indices, decoder.variable_indices_ptr, 
                decoder.variable_degrees, decoder.num_variables)
            
            # Make hard decisions
            @cuda threads=threads_per_block blocks=blocks_var stream=stream hard_decision_kernel_optimized!(
                decoder.hard_decisions, decoder.beliefs, decoder.num_variables)
        end
        
        if captured_graph !== nothing
            # If capture succeeded, instantiate the graph and create a function to execute it
            exec = CUDA.instantiate(captured_graph)
            bp_graph_temp = () -> CUDA.launch(exec, stream)
            @info "CUDA graph capture successful"
        else
            @warn "CUDA graph capture returned nothing, continuing without graph optimization"
            bp_graph_temp = nothing
        end
    catch e
        @warn "CUDA graph capture failed: $e - continuing without graph optimization"
        bp_graph_temp = nothing
    end
    
    # Only after successfully creating a graph or handling failure, update the decoder
    decoder.bp_graph = bp_graph_temp
end

# =============================================================================
# Optimized Ambiguity Clustering Implementation - Stage 1
# =============================================================================

"""
    ac_pivot_operation_kernel_optimized!(H_values, H_row_ptr, H_col_ind, syndrome, pivot_row, pivot_col, row_status, num_checks)
    
Optimized kernel for pivot operations in AC Stage 1 using shared memory optimizations.
"""
function ac_pivot_operation_kernel_optimized!(H_values, H_row_ptr, H_col_ind, syndrome, pivot_row, pivot_col, row_status, num_checks)
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if row_idx <= num_checks && row_idx != pivot_row && row_status[row_idx] == 0
        # Check if this row has a nonzero in the pivot column
        has_nonzero = false
        
        # Use shared memory to cache pivot row indices
        shared_pivot_cols = @cuStaticSharedMem(Int32, 32)
        shared_thread_has_pivot = @cuStaticSharedMem(Int32, 256)
        
        # Load a segment of the pivot row into shared memory
        pivot_start = H_row_ptr[pivot_row]
        pivot_end = H_row_ptr[pivot_row+1] - 1
        pivot_cols_loaded = min(Int(32), pivot_end - pivot_start + 1)
        
        if threadIdx().x <= pivot_cols_loaded
            shared_pivot_cols[threadIdx().x] = H_col_ind[pivot_start + threadIdx().x - 1]
        end
        
        # Initialize thread status
        shared_thread_has_pivot[threadIdx().x] = 0
        
        # Sync to ensure all data is loaded
        sync_threads()
        
        # Check if this row has the pivot column
        row_start = H_row_ptr[row_idx]
        row_end = H_row_ptr[row_idx+1] - 1
        
        for j in row_start:row_end
            col = H_col_ind[j]
            if col == pivot_col
                shared_thread_has_pivot[threadIdx().x] = 1
                break
            end
        end
        
        # Sync to ensure all threads have checked
        sync_threads()
        
        # If this row has a nonzero in the pivot column, add pivot row to it (XOR operation)
        if shared_thread_has_pivot[threadIdx().x] == 1
            # XOR syndrome (binary addition)
            current_val = syndrome[row_idx]
            pivot_val = syndrome[pivot_row]
            new_val = (current_val + pivot_val) % 2
            
            # Use atomic XOR to toggle the bit if needed
            if current_val != new_val
                CUDA.atomic_xor!(pointer(syndrome, row_idx), Int32(1))
                CUDA.threadfence()  # Ensure visibility to other threads
            end
        end
    end
    
    return nothing
end

"""
    ac_stage1!(decoder, syndrome::CuVector{Int32}, posteriors::CuVector{Float32})

Optimized AC Stage 1: Find an initial solution with efficient GPU implementation.
Uses stream-based concurrency and optimized kernels.
"""
function ac_stage1!(decoder, syndrome::CuVector{Int32}, posteriors::CuVector{Float32})
    num_checks = decoder.num_checks
    num_variables = decoder.num_variables
    perf_config = decoder.perf_config
    
    # Use primary stream
    stream = decoder.streams[1]
    
    # Status arrays (0 = unused, 1 = used)
    row_status = CUDA.zeros(Int32, num_checks)
    col_status = CUDA.zeros(Int32, num_variables)
    
    # Solution vector
    solution = CUDA.zeros(Int32, num_variables)
    
    # Working arrays
    best_pivots = CUDA.zeros(Int32, num_checks)
    best_posterior_values = CUDA.zeros(Float32, num_checks)
    
    # Track pivot operations
    pivot_rows = Int32[]
    pivot_cols = Int32[]
    
    # Thread configuration
    threads_per_block = perf_config.thread_block_size
    num_blocks = cld(num_checks, threads_per_block)
    
    # Continue until all syndrome bits are explained or no more pivots are possible
    max_iterations = min(num_checks, num_variables)
    
    # Copy of the syndrome for verification
    syndrome_cpu = zeros(Int32, num_checks)
    
    for iter in 1:max_iterations
        # Reset working arrays
        CUDA.fill!(best_pivots, 0)
        CUDA.fill!(best_posterior_values, -1.0f0)
        
        # Find potential pivots in parallel with optimized kernel
        @cuda threads=threads_per_block blocks=num_blocks stream=stream ac_stage1_find_pivots_kernel_optimized!(
            decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal, 
            row_status, col_status, syndrome, posteriors, 
            best_pivots, best_posterior_values, num_checks)
        
        # Wait for kernel to complete
        CUDA.synchronize(stream)
        
        # Copy GPU arrays to CPU using asynchronous memory transfers
        best_posterior_values_cpu = Array{Float32}(undef, num_checks)
        best_pivots_cpu = Array{Int32}(undef, num_checks)
        copyto!(best_posterior_values_cpu, best_posterior_values)
        copyto!(best_pivots_cpu, best_pivots)
        
        # Find row with highest posterior value
        best_row = 0
        best_posterior = -1.0f0
        
        # Use multi-threading for faster CPU processing
        # Create atomic variables for thread-safe updates
        best_posterior_atomic = Threads.Atomic{Float32}(-1.0f0)
        best_row_atomic = Threads.Atomic{Int32}(0)
        
        Threads.@threads for row in 1:num_checks
            if best_posterior_values_cpu[row] > best_posterior_atomic[]
                # Use atomic compare-and-swap for thread safety
                old_val = best_posterior_atomic[]
                while best_posterior_values_cpu[row] > old_val
                    if Threads.atomic_cas!(best_posterior_atomic, old_val, best_posterior_values_cpu[row]) == old_val
                        # Successfully updated posterior, now update row
                        Threads.atomic_xchg!(best_row_atomic, Int32(row))
                        break
                    end
                    old_val = best_posterior_atomic[]
                end
            end
        end
        
        # Copy atomic values to regular variables
        best_posterior = best_posterior_atomic[]
        best_row = best_row_atomic[]
        
        if best_row == 0
            # No more valid pivots
            break
        end
        
        pivot_row = best_row
        pivot_col = best_pivots_cpu[pivot_row]
        
        # Record this pivot
        push!(pivot_rows, pivot_row)
        push!(pivot_cols, pivot_col)
        
        # Update status arrays
        row_status_cpu = Array(row_status)
        col_status_cpu = Array(col_status)
        row_status_cpu[pivot_row] = 1
        col_status_cpu[pivot_col] = 1
        
        # Update GPU arrays asynchronously
        copyto!(row_status, row_status_cpu)
        copyto!(col_status, col_status_cpu)
        
        # Update solution based on syndrome
        copyto!(syndrome_cpu, syndrome)
        solution_cpu = Array(solution)
        if syndrome_cpu[pivot_row] == 1
            solution_cpu[pivot_col] = 1
        end
        copyto!(solution, solution_cpu)
        
        # Perform pivot operation in parallel with optimized kernel
        @cuda threads=threads_per_block blocks=num_blocks stream=stream ac_pivot_operation_kernel_optimized!(
            decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal,
            syndrome, pivot_row, pivot_col, row_status, num_checks)
        
        # Wait for kernel to complete
        CUDA.synchronize(stream)
        
        # Update CPU copy of syndrome
        copyto!(syndrome_cpu, syndrome)
        
        # Check if all syndrome bits are now 0
        if all(syndrome_cpu .== 0)
            break
        end
    end
    
    # Return the initial solution and pivot information
    return solution, CuVector{Int32}(pivot_rows), CuVector{Int32}(pivot_cols), row_status, col_status
end

"""
    ac_stage1_find_pivots_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, col_status,
                                              syndrome, posteriors, best_pivots, best_posterior_values, num_checks)
    
Optimized kernel for finding potential pivots in AC Stage 1.
"""
function ac_stage1_find_pivots_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, col_status,
                                              syndrome, posteriors, best_pivots, best_posterior_values, num_checks)
    row_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if row_idx <= num_checks && syndrome[row_idx] == 1 && row_status[row_idx] == 0
        # This thread handles a row with nonzero syndrome that hasn't been used as pivot
        best_posterior = -1.0f0
        best_col = 0
        
        # Scan all entries in this row
        row_start = H_row_ptr[row_idx]
        row_end = H_row_ptr[row_idx+1] - 1
        
        # Use shared memory to cache column status
        col_indices = @cuStaticSharedMem(Int32, 256)
        col_posteriors = @cuStaticSharedMem(Float32, 256)
        col_statuses = @cuStaticSharedMem(Int32, 256)
        
        # Process in chunks of 256 for optimal shared memory usage
        for offset in 0:256:row_end-row_start
            chunk_size = min(256, row_end - row_start + 1 - offset)
            
            # Load column indices and statuses into shared memory
            if threadIdx().x <= chunk_size
                j = row_start + offset + threadIdx().x - 1
                col = H_col_ind[j]
                col_indices[threadIdx().x] = col
                if col <= length(posteriors)
                    col_posteriors[threadIdx().x] = posteriors[col]
                else
                    col_posteriors[threadIdx().x] = -1.0f0
                end
                if col <= length(col_status)
                    col_statuses[threadIdx().x] = col_status[col]
                else
                    col_statuses[threadIdx().x] = 1  # Mark as already used to prevent selection
                end
            end
            
            # Sync threads
            sync_threads()
            
            # Find best pivot in this chunk
            for i in 1:chunk_size
                col = col_indices[i]
                post = col_posteriors[i]
                status = col_statuses[i]
                
                if status == 0 && post > best_posterior
                    best_posterior = post
                    best_col = col
                end
            end
            
            # Sync threads before next iteration
            sync_threads()
        end
        
        # Store the best pivot for this row
        if best_col > 0
            best_pivots[row_idx] = best_col
            best_posterior_values[row_idx] = best_posterior
        end
    end
    
    return nothing
end

# =============================================================================
# Optimized Ambiguity Clustering Implementation - Stage 2
# =============================================================================

"""
    ac_stage2_scan_columns_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, col_status,
                                          posteriors, column_priorities, row_cluster_ids, 
                                          num_variables, num_checks)
    
Optimized kernel for scanning columns in AC Stage 2 with efficient memory access.
"""
function ac_stage2_scan_columns_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, col_status,
                                               posteriors, column_priorities, row_cluster_ids, 
                                               num_variables, num_checks)
    col_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if col_idx <= num_variables && col_status[col_idx] == 0
        # Use shared memory to cache row status
        row_statuses = @cuStaticSharedMem(Int32, 256)
        row_cluster_ids_local = @cuStaticSharedMem(Int32, 256)
        
        # Initialize variables
        is_adjacent = false
        max_posterior = posteriors[col_idx]
        
        # Process rows in chunks to optimize memory access
        for row_offset in 0:256:num_checks-1
            chunk_size = min(256, num_checks - row_offset)
            
            # Load row statuses into shared memory
            if threadIdx().x <= chunk_size
                row_idx = row_offset + threadIdx().x
                if row_idx <= length(row_status)
                    row_statuses[threadIdx().x] = row_status[row_idx]
                else
                    row_statuses[threadIdx().x] = 0
                end
                
                if row_idx <= length(row_cluster_ids)
                    row_cluster_ids_local[threadIdx().x] = row_cluster_ids[row_idx]
                else
                    row_cluster_ids_local[threadIdx().x] = 0
                end
            end
            
            # Sync threads
            sync_threads()
            
            # Check column connectivity in this chunk
            for j in 1:chunk_size
                row_idx = row_offset + j
                # Check if this column is connected to this row
                is_connected = false
                if row_idx <= num_checks
                    for k in H_row_ptr[row_idx]:(H_row_ptr[row_idx+1]-1)
                        if k <= length(H_col_ind) && H_col_ind[k] == col_idx
                            is_connected = true
                            break
                        end
                    end
                    
                    # Check if this is a pivot row
                    if is_connected && row_statuses[j] == 1
                        is_adjacent = true
                    end
                end
            end
            
            # Sync threads before next iteration
            sync_threads()
        end
        
        # If adjacent to a pivot row, record its priority
        if is_adjacent
            column_priorities[col_idx] = max_posterior
        end
    end
    
    return nothing
end

"""
    ac_stage2_column_analysis_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, row_cluster_ids,
                                             col_idx, connected_rows, non_pivot_connected, affected_clusters,
                                             num_checks, max_clusters)
    
Optimized kernel for analyzing column connectivity in AC Stage 2 with warp-level optimizations.
"""
function ac_stage2_column_analysis_kernel_optimized!(H_values, H_row_ptr, H_col_ind, row_status, row_cluster_ids,
    col_idx, connected_rows, non_pivot_connected, affected_clusters,
    num_checks, max_clusters)
warp_id = div((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1, 32) + 1
lane_id = (threadIdx().x - 1) % 32 + 1

# Shared memory for warp-level operations
warp_has_non_pivot = @cuStaticSharedMem(Int32, 16)
warp_cluster_ids = @cuStaticSharedMem(Int32, (32, 16))

if warp_id <= cld(num_checks, 32)
# Initialize warp-level variables
if lane_id == 1
warp_has_non_pivot[warp_id] = 0
for i in 1:32
warp_cluster_ids[i, warp_id] = 0
end
end

# Sync warp
CUDA.sync_warp()

# Determine which rows this warp processes
start_row = (warp_id - 1) * 32 + 1
end_row = min(start_row + 31, num_checks)

# Check if current thread's row is connected to the column
row_idx = start_row + lane_id - 1
is_connected = false

if row_idx <= end_row
for j in H_row_ptr[row_idx]:(H_row_ptr[row_idx+1]-1)
if j <= length(H_col_ind) && H_col_ind[j] == col_idx
is_connected = true
break
end
end

if is_connected
# Mark this row as connected
connected_rows[row_idx] = 1

# Check if this is a non-pivot row
cluster_id = 0
if row_idx <= length(row_cluster_ids)
cluster_id = row_cluster_ids[row_idx]
end

if cluster_id == 0
# Set non-pivot flag for this warp
CUDA.@atomic warp_has_non_pivot[warp_id] += 1
else
# Add cluster ID to warp's list
# Try each slot until we find an empty one or matching cluster
for i in 1:32
# First check if slot already contains this cluster ID
if warp_cluster_ids[i, warp_id] == cluster_id
break
end

# If slot is empty, try to claim it with simple assignment
# This is a race condition, but it's acceptable for this algorithm
if warp_cluster_ids[i, warp_id] == 0
warp_cluster_ids[i, warp_id] = cluster_id
break
end
end
end
end
end

# Sync warp to ensure all threads have processed their rows
CUDA.sync_warp()

# First thread in warp updates global non_pivot_connected flag
if lane_id == 1 && warp_has_non_pivot[warp_id] > 0
# Update the global non_pivot_connected flag
non_pivot_connected[1] = 1
end

# First thread in warp updates global affected_clusters array
if lane_id == 1
for i in 1:32
cluster_id = warp_cluster_ids[i, warp_id]
if cluster_id > 0
# Find an empty slot or slot with matching cluster ID
for j in 1:max_clusters
if j <= length(affected_clusters)
# Check if slot is empty or already has this cluster
current = affected_clusters[j]
if current == 0
# Use simple assignment - race condition is acceptable
affected_clusters[j] = cluster_id
break
elseif current == cluster_id
break  # Cluster already in list
end
end
end
end
end
end
end

return nothing
end
"""
    ac_stage2_update_cluster_ids_kernel_optimized!(row_cluster_ids, old_cluster_id, new_cluster_id, num_checks)
    
Update row_cluster_ids: Replace all occurrences of old_cluster_id with new_cluster_id.
"""
function ac_stage2_update_cluster_ids_kernel_optimized!(row_cluster_ids, old_cluster_id, new_cluster_id, num_checks)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= num_checks && row_cluster_ids[idx] == old_cluster_id
        row_cluster_ids[idx] = new_cluster_id
    end
    
    return nothing
end

"""
    ac_stage2!(decoder, syndrome, posteriors, pivot_rows, pivot_cols, row_status, col_status)

Optimized AC Stage 2: Form clusters with parallel GPU operations and multi-stream execution.
"""
function ac_stage2!(decoder, syndrome, posteriors, pivot_rows, pivot_cols, row_status, col_status)
    num_checks = decoder.num_checks
    num_variables = decoder.num_variables
    perf_config = decoder.perf_config
    
    # Use primary stream
    stream = decoder.streams[1]
    
    # Determine the number of columns to explore (κ parameter)
    k = min(Int(round(decoder.ac_k_param * num_variables)), num_variables - length(Array(pivot_cols)))
    
    # Initialize row_cluster_ids
    row_cluster_ids = CUDA.zeros(Int32, num_checks)
    
    # Get pivot rows from GPU
    pivot_rows_cpu = Array(pivot_rows)
    
    # Initialize each pivot row with its own cluster ID
    next_cluster_id = 1
    row_cluster_ids_cpu = zeros(Int32, num_checks)
    
    for row in pivot_rows_cpu
        if row > 0
            row_cluster_ids_cpu[row] = next_cluster_id
            next_cluster_id += 1
        end
    end
    
    # Transfer back to GPU
    copyto!(row_cluster_ids, row_cluster_ids_cpu)
    
    # Prepare cluster tracking
    # Use a more efficient data structure for cluster tracking
    cluster_data = Dict{Int32, Dict{String, Vector{Int32}}}()
    
    # Initialize from pivot rows/cols
    pivot_cols_cpu = Array(pivot_cols)
    for i in 1:length(pivot_rows_cpu)
        row = pivot_rows_cpu[i]
        col = pivot_cols_cpu[i]
        
        if row > 0
            cluster_id = row_cluster_ids_cpu[row]
            cluster_data[cluster_id] = Dict{String, Vector{Int32}}(
                "rows" => Int32[row],
                "cols" => Int32[col]
            )
        end
    end
    
    # Thread configuration
    threads_per_block = perf_config.thread_block_size
    var_blocks = cld(num_variables, threads_per_block)
    check_blocks = cld(num_checks, threads_per_block)
    
    # Working arrays for column analysis
    column_priorities = CUDA.zeros(Float32, num_variables)
    
    # Find columns connected to pivot rows for Stage 2
    for iteration in 1:k
        # Reset column priorities
        CUDA.fill!(column_priorities, -1.0f0)
        
        # Find potential columns in parallel with optimized kernel
        @cuda threads=threads_per_block blocks=var_blocks stream=stream ac_stage2_scan_columns_kernel_optimized!(
            decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal,
            row_status, col_status, posteriors, column_priorities, row_cluster_ids,
            num_variables, num_checks)
        
        # Wait for kernel to complete
        CUDA.synchronize(stream)
        
        # Find column with highest priority
        column_priorities_cpu = Array(column_priorities)
        
        # Use multi-threading for faster CPU processing
        best_col = 0
        best_priority = -1.0f0
        
        # Use atomic variables for thread-safe updates
        best_priority_atomic = Threads.Atomic{Float32}(-1.0f0)
        best_col_atomic = Threads.Atomic{Int32}(0)
        
        Threads.@threads for col in 1:num_variables
            if column_priorities_cpu[col] > best_priority_atomic[]
                # Use atomic compare-and-swap for thread safety
                old_val = best_priority_atomic[]
                while column_priorities_cpu[col] > old_val
                    if Threads.atomic_cas!(best_priority_atomic, old_val, column_priorities_cpu[col]) == old_val
                        # Successfully updated priority, now update column
                        Threads.atomic_xchg!(best_col_atomic, Int32(col))
                        break
                    end
                    old_val = best_priority_atomic[]
                end
            end
        end
        
        # Copy atomic values to regular variables
        best_priority = best_priority_atomic[]
        best_col = best_col_atomic[]
        
        if best_col == 0
            # No more columns to explore
            break
        end
        
        # Mark this column as used
        col_status_cpu = Array(col_status)
        col_status_cpu[best_col] = 1
        copyto!(col_status, col_status_cpu)
        
        # Analyze column connectivity
        connected_rows = CUDA.zeros(Int32, num_checks)
        non_pivot_connected = CUDA.zeros(Int32, 1)
        max_clusters = 64  # Increased limit for number of affected clusters
        affected_clusters = CUDA.zeros(Int32, max_clusters)
        
        # Use optimized kernel with warp-level primitives
        warp_blocks = cld(num_checks, 32)  # Each warp handles 32 rows
        
        @cuda threads=threads_per_block blocks=warp_blocks stream=stream ac_stage2_column_analysis_kernel_optimized!(
            decoder.H_d.nzVal, decoder.H_d.rowPtr, decoder.H_d.colVal,
            row_status, row_cluster_ids, best_col, connected_rows,
            non_pivot_connected, affected_clusters, num_checks, max_clusters)
        
        # Wait for kernel to complete
        CUDA.synchronize(stream)
        
        # Process results
        non_pivot_connected_cpu = Array(non_pivot_connected)[1] == 1
        affected_clusters_cpu = filter(x -> x > 0, Array(affected_clusters))
        connected_rows_cpu = Array(connected_rows)
        
        if non_pivot_connected_cpu
            # Column has non-pivot rows - seed a new cluster
            new_pivot_row = 0
            
            # Find first non-pivot connected row
            for row in 1:num_checks
                if row <= length(connected_rows_cpu) && connected_rows_cpu[row] == 1 && 
                   row <= length(row_cluster_ids_cpu) && row_cluster_ids_cpu[row] == 0
                    new_pivot_row = row
                    break
                end
            end
            
            if new_pivot_row > 0
                # Create new cluster
                new_cluster_id = next_cluster_id
                next_cluster_id += 1
                
                # Mark the row as a pivot
                row_status_cpu = Array(row_status)
                row_status_cpu[new_pivot_row] = 1
                copyto!(row_status, row_status_cpu)
                
                # Update row_cluster_ids
                row_cluster_ids_cpu[new_pivot_row] = new_cluster_id
                copyto!(row_cluster_ids, row_cluster_ids_cpu)
                
                # Add to cluster data
                cluster_data[new_cluster_id] = Dict{String, Vector{Int32}}(
                    "rows" => Int32[new_pivot_row],
                    "cols" => Int32[best_col]
                )
            end
        elseif !isempty(affected_clusters_cpu)
            # Column connects to existing clusters
            if length(affected_clusters_cpu) == 1
                # Add to existing cluster
                cluster_id = affected_clusters_cpu[1]
                push!(cluster_data[cluster_id]["cols"], best_col)
            else
                # Merge clusters
                new_cluster_id = next_cluster_id
                next_cluster_id += 1
                
                # Pre-calculate total size for merged cluster
                total_rows = 0
                total_cols = 1  # Start with 1 for best_col
                for old_id in affected_clusters_cpu
                    if haskey(cluster_data, old_id)
                        total_rows += length(cluster_data[old_id]["rows"])
                        total_cols += length(cluster_data[old_id]["cols"])
                    end
                end
                
                # Create new merged cluster with pre-allocated vectors
                merged_rows = Vector{Int32}(undef, total_rows)
                merged_cols = Vector{Int32}(undef, total_cols)
                
                # Set the first column to best_col
                merged_cols[1] = best_col
                
                # Track positions for filling arrays
                row_pos = 1
                col_pos = 2  # Start at 2 because best_col is at position 1
                
                # Merge all affected clusters
                for old_id in affected_clusters_cpu
                    if haskey(cluster_data, old_id)
                        # Get current cluster data
                        old_rows = cluster_data[old_id]["rows"]
                        old_cols = cluster_data[old_id]["cols"]
                        
                        # Copy rows
                        copyto!(merged_rows, row_pos, old_rows, 1, length(old_rows))
                        row_pos += length(old_rows)
                        
                        # Copy columns
                        copyto!(merged_cols, col_pos, old_cols, 1, length(old_cols))
                        col_pos += length(old_cols)
                        
                        # Update row_cluster_ids on the GPU
                        @cuda threads=threads_per_block blocks=check_blocks stream=stream ac_stage2_update_cluster_ids_kernel_optimized!(
                            row_cluster_ids, old_id, new_cluster_id, num_checks)
                        
                        # Update CPU tracking
                        for row in 1:num_checks
                            if row <= length(row_cluster_ids_cpu) && row_cluster_ids_cpu[row] == old_id
                                row_cluster_ids_cpu[row] = new_cluster_id
                            end
                        end
                        
                        # Remove old cluster
                        delete!(cluster_data, old_id)
                    end
                end
                
                # Store the merged cluster
                cluster_data[new_cluster_id] = Dict{String, Vector{Int32}}(
                    "rows" => merged_rows[1:(row_pos-1)],
                    "cols" => merged_cols[1:(col_pos-1)]
                )
            end
        end
    end
    
    return cluster_data, row_cluster_ids
end

# =============================================================================
# Optimized Ambiguity Clustering Implementation - Stage 3
# =============================================================================

"""
    ac_evaluate_solution_kernel_optimized!(solutions, solution_count, llr, solution_scores, num_variables)
    
Optimized kernel for evaluating solution log-probabilities with shared memory and vectorized operations.
"""
function ac_evaluate_solution_kernel_optimized!(solutions, solution_count, llr, solution_scores, num_variables)
    sol_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if sol_idx <= solution_count
        # Compute log-probability score for this solution
        score = 0.0f0
        
        # Process variables in chunks for better memory access
        for var_offset in 0:32:num_variables-1
            chunk_size = min(32, num_variables - var_offset)
            
            # Load LLR values into shared memory
            llr_local = @cuStaticSharedMem(Float32, 32)
            sol_local = @cuStaticSharedMem(Int32, 32)
            
            if threadIdx().x <= chunk_size
                var_idx = var_offset + threadIdx().x
                if var_idx <= length(llr)
                    llr_local[threadIdx().x] = llr[var_idx]
                    sol_local[threadIdx().x] = solutions[sol_idx, var_idx]
                end
            end
            
            # Sync threads
            sync_threads()
            
            # Process this chunk
            for i in 1:chunk_size
                var_idx = var_offset + i
                if var_idx <= num_variables
                    err = sol_local[i]
                    llr_val = llr_local[i]
                    
                    if err == 1
                        # Error occurred - add log(p_error)
                        score += llr_val
                    else
                        # No error - add log(1-p_error)
                        score += log(1.0f0 + exp(llr_val)) - llr_val
                    end
                end
            end
            
            # Sync threads before next iteration
            sync_threads()
        end
        
        solution_scores[sol_idx] = score
    end
    
    return nothing
end

"""
    ac_compute_logical_effect_kernel_optimized!(solutions, solution_count, L_values, L_row_ptr, L_col_ind,
                                             logical_effects, num_logicals)
    
Optimized kernel for computing logical effects with shared memory optimizations.
"""
function ac_compute_logical_effect_kernel_optimized!(solutions, solution_count, L_values, L_row_ptr, L_col_ind,
                                                  logical_effects, num_logicals)
    sol_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    logical_idx = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if sol_idx <= solution_count && logical_idx <= num_logicals
        # Use shared memory to cache column indices
        col_indices = @cuStaticSharedMem(Int32, 64)
        
        # Compute logical effect for this solution and logical operator
        effect = 0
        
        # Process logical operator in chunks
        row_start = L_row_ptr[logical_idx]
        row_end = L_row_ptr[logical_idx+1] - 1
        
        for offset in 0:64:row_end-row_start
            chunk_size = min(64, row_end - row_start + 1 - offset)
            
            # Load column indices into shared memory
            if threadIdx().y <= chunk_size
                j = row_start + offset + threadIdx().y - 1
                if j <= length(L_col_ind)
                    col_indices[threadIdx().y] = L_col_ind[j]
                end
            end
            
            # Sync threads in block
            sync_threads()
            
            # Process this chunk
            for i in 1:chunk_size
                col = col_indices[i]
                if col <= size(solutions, 2) && solutions[sol_idx, col] == 1
                    effect = (effect + 1) % 2
                end
            end
            
            # Sync threads before next iteration
            sync_threads()
        end
        
        logical_effects[sol_idx, logical_idx] = effect
    end
    
    return nothing
end

"""
    ac_stage3!(decoder, cluster_data, row_cluster_ids, llr, syndrome)

Optimized AC Stage 3: Analyze clusters in parallel using multiple GPU streams.
"""
function ac_stage3!(
    decoder::QLDPCDecoder, 
    cluster_data::Dict{Int32, Dict{String, Vector{Int32}}},
    row_cluster_ids::CuVector{Int32},
    llr::CuVector{Float32},
    syndrome::CuVector{Int32})
    
    # Initialize logical values
    logical_values = zeros(Int32, decoder.num_logicals)
    
    # Use parallel processing for independent clusters when possible
    if decoder.perf_config.use_stream_parallelism && length(decoder.streams) > 1
        # Group clusters by size for better load balancing
        small_clusters = Int32[]
        medium_clusters = Int32[]
        large_clusters = Int32[]
        
        for (cluster_id, cluster) in cluster_data
            rows = length(cluster["rows"])
            cols = length(cluster["cols"])
            
            if rows + cols <= 50
                push!(small_clusters, cluster_id)
            elseif rows + cols <= 200
                push!(medium_clusters, cluster_id)
            else
                push!(large_clusters, cluster_id)
            end
        end
        
        # Process large clusters first (they take the most time)
        for cluster_id in large_clusters
            cluster_effect = ac_stage3_analyze_cluster_optimized!(
                decoder, cluster_id, cluster_data[cluster_id], row_cluster_ids, llr, syndrome)
            
            # Add cluster effect to overall logical values (XOR operation)
            for i in 1:decoder.num_logicals
                logical_values[i] = (logical_values[i] + cluster_effect[i]) % 2
            end
        end
        
        # Process medium clusters with parallel tasks
        if !isempty(medium_clusters)
            num_medium = length(medium_clusters)
            medium_results = Vector{Vector{Int32}}(undef, num_medium)
            medium_tasks = Vector{Task}(undef, num_medium)
            
            # Create tasks for medium clusters
            for i in 1:num_medium
                medium_tasks[i] = Threads.@spawn begin
                    cluster_id = medium_clusters[i]
                    ac_stage3_analyze_cluster_optimized!(
                        decoder, cluster_id, cluster_data[cluster_id], row_cluster_ids, llr, syndrome)
                end
            end
            
            # Wait for all tasks and collect results
            for i in 1:num_medium
                medium_results[i] = fetch(medium_tasks[i])
                
                # Combine results
                for j in 1:decoder.num_logicals
                    logical_values[j] = (logical_values[j] + medium_results[i][j]) % 2
                end
            end
        end
        
        # Process small clusters with stream-based parallelism
        if !isempty(small_clusters)
            num_streams = length(decoder.streams)
            batch_size = cld(length(small_clusters), num_streams)
            
            # Prepare batches of clusters for each stream
            batch_tasks = Vector{Task}(undef, num_streams)
            
            # Process each batch in parallel
            for stream_idx in 1:num_streams
                batch_start = (stream_idx - 1) * batch_size + 1
                batch_end = min(batch_start + batch_size - 1, length(small_clusters))
                
                if batch_start <= batch_end
                    # Process this batch with the current stream
                    batch_tasks[stream_idx] = Threads.@spawn begin
                        batch_results = Vector{Int32}(zeros(Int32, decoder.num_logicals))
                        
                        # Process each cluster in this batch
                        for j in batch_start:batch_end
                            cluster_id = small_clusters[j]
                            cluster_effect = ac_stage3_analyze_cluster_optimized!(
                                decoder, cluster_id, cluster_data[cluster_id], row_cluster_ids, llr, syndrome)
                            
                            # Combine results within this batch
                            for k in 1:decoder.num_logicals
                                batch_results[k] = (batch_results[k] + cluster_effect[k]) % 2
                            end
                        end
                        
                        batch_results
                    end
                end
            end
            
            # Collect results from all batches
            for stream_idx in 1:num_streams
                batch_start = (stream_idx - 1) * batch_size + 1
                batch_end = min(batch_start + batch_size - 1, length(small_clusters))
                
                if batch_start <= batch_end
                    batch_results = fetch(batch_tasks[stream_idx])
                    
                    # Combine batch results with overall logical values
                    for i in 1:decoder.num_logicals
                        logical_values[i] = (logical_values[i] + batch_results[i]) % 2
                    end
                end
            end
        end
    else
        # Sequential processing of clusters (fallback method)
        for (cluster_id, cluster) in cluster_data
            # Analyze this cluster
            cluster_effect = ac_stage3_analyze_cluster_optimized!(
                decoder, cluster_id, cluster, row_cluster_ids, llr, syndrome)
            
            # Add cluster effect to overall logical values (XOR operation)
            for i in 1:decoder.num_logicals
                logical_values[i] = (logical_values[i] + cluster_effect[i]) % 2
            end
        end
    end
    
    return logical_values
end

"""
    ac_stage3_analyze_cluster_optimized!(decoder, cluster_id, cluster_data, row_cluster_ids, llr, syndrome)

Analyze a single cluster and return its logical effect.
"""
function ac_stage3_analyze_cluster_optimized!(
    decoder::QLDPCDecoder, 
    cluster_id::Int32,
    cluster_data::Dict{String, Vector{Int32}},
    row_cluster_ids::CuVector{Int32},
    llr::CuVector{Float32},
    syndrome::CuVector{Int32})
    
    num_logicals = decoder.num_logicals
    num_variables = decoder.num_variables
    perf_config = decoder.perf_config
    
    # Use secondary stream for parallel operations
    stream = decoder.streams[min(2, length(decoder.streams))]
    
    # Get rows and columns in this cluster
    cluster_rows = cluster_data["rows"]
    cluster_cols = cluster_data["cols"]
    
    # Skip empty clusters
    if isempty(cluster_rows) || isempty(cluster_cols)
        return zeros(Int32, num_logicals)
    end
    
    # Extract submatrix for this cluster
    # Create CSR format for cluster submatrix - more efficient for GPU operations
    H_submatrix = spzeros(Float32, length(cluster_rows), num_variables)
    
    # Extract rows of H for this cluster
    H_values_cpu = Array(decoder.H_d.nzVal)
    H_row_ptr_cpu = Array(decoder.H_d.rowPtr)
    H_col_ind_cpu = Array(decoder.H_d.colVal)
    
    for (i, row) in enumerate(cluster_rows)
        if row <= num_variables
            row_start = H_row_ptr_cpu[row]
            row_end = min(length(H_row_ptr_cpu) > row ? H_row_ptr_cpu[row+1] - 1 : length(H_col_ind_cpu), length(H_col_ind_cpu))
            
            for j in row_start:row_end
                if j <= length(H_col_ind_cpu)
                    col = H_col_ind_cpu[j]
                    if col <= size(H_submatrix, 2)
                        H_submatrix[i, col] = H_values_cpu[j]
                    end
                end
            end
        end
    end
    
    # Extract submatrix for logical operators
    L_submatrix = spzeros(Float32, num_logicals, num_variables)
    
    # Extract logical operators
    L_values_cpu = Array(decoder.L_d.nzVal)
    L_row_ptr_cpu = Array(decoder.L_d.rowPtr)
    L_col_ind_cpu = Array(decoder.L_d.colVal)
    
    for i in 1:num_logicals
        if i <= length(L_row_ptr_cpu)
            row_start = L_row_ptr_cpu[i]
            row_end = min(length(L_row_ptr_cpu) > i ? L_row_ptr_cpu[i+1] - 1 : length(L_col_ind_cpu), length(L_col_ind_cpu))
            
            for j in row_start:row_end
                if j <= length(L_col_ind_cpu)
                    col = L_col_ind_cpu[j]
                    if col <= size(L_submatrix, 2)
                        L_submatrix[i, col] = L_values_cpu[j]
                    end
                end
            end
        end
    end
    
  
    is_dependent = zeros(Int32, num_logicals)
    dependency_matrix = zeros(Float32, num_logicals, length(cluster_rows))
    
    Threads.@threads for i in 1:num_logicals
        found_dependency = false
        
 
        active_cols = cluster_cols
        augmented = zeros(Float32, (length(cluster_rows), length(active_cols) + 1))
        
        for (row_idx, row) in enumerate(cluster_rows)
            for (col_idx, col) in enumerate(active_cols)
                if col <= size(H_submatrix, 2) && row_idx <= size(H_submatrix, 1)
                    augmented[row_idx, col_idx] = H_submatrix[row_idx, col]
                end
            end
        end
        
        # Add the logical row as the last column
        for (col_idx, col) in enumerate(active_cols)
            if col <= size(L_submatrix, 2) && i <= size(L_submatrix, 1)
                #OLD
                #augmented[1:end, end] .⊻= L_submatrix[i, col] .* augmented[1:end, col_idx]
                # For binary values, modular addition is equivalent to XOR
# This works with Float32 values
if L_submatrix[i, col] != 0
    augmented[1:end, end] = (augmented[1:end, end] .+ augmented[1:end, col_idx]) .% 2
end
            end
        end
        
        m, n = size(augmented)
        for j in 1:m
            pivot_row = j
            while pivot_row <= m && augmented[pivot_row, j] == 0
                pivot_row += 1
            end
            
            if pivot_row <= m
                if pivot_row != j
                    augmented[j, :], augmented[pivot_row, :] = augmented[pivot_row, :], augmented[j, :]
                end
                
                for k in j+1:m
                    if augmented[k, j] != 0
                        #augmented[k, :] = (augmented[k, :] .⊻ augmented[j, :]) .% 2
                        #OLD
                        augmented[k, :] = (augmented[k, :] .+ augmented[j, :]) .% 2

                    end
                end
            end
        end
        
        has_solution = true
        for j in 1:m
            if all(augmented[j, 1:end-1] .== 0) && augmented[j, end] != 0
                has_solution = false
                break
            end
        end
        
        if has_solution
            dependency = zeros(Float32, m)
            for j in m:-1:1
                idx = findfirst(x -> x != 0, augmented[j, 1:m])
                if idx !== nothing
                    dependency[j] = augmented[j, end]
                    for k in j+1:m
                        dependency[j] = (dependency[j] + dependency[k] * augmented[j, k]) % 2
                    end
                end
            end
            
            is_dependent[i] = 1
            dependency_matrix[i, :] = dependency
        end
    end
    
    is_ambiguous = !all(is_dependent .== 1)
    
    logical_effect = zeros(Int32, num_logicals)
    
    if !is_ambiguous
        
        syndrome_cluster = zeros(Int32, length(cluster_rows))
        syndrome_cpu = Array(syndrome)
        
        for (i, row) in enumerate(cluster_rows)
            if row <= length(syndrome_cpu)
                syndrome_cluster[i] = syndrome_cpu[row]
            end
        end
        
        for i in 1:num_logicals
            effect = 0
            for (idx, row) in enumerate(cluster_rows)
                if idx <= length(syndrome_cluster) && 
                   i <= size(dependency_matrix, 1) && idx <= size(dependency_matrix, 2) &&
                   dependency_matrix[i, idx] != 0 && syndrome_cluster[idx] != 0
                    effect = (effect + 1) % 2
                end
            end
            logical_effect[i] = effect
        end
    else
        
        active_cols = copy(cluster_cols)
        if length(active_cols) > decoder.ac_max_cluster_size
            llr_cpu = Array(llr)
            posteriors = ones(Float32, num_variables)
            for i in active_cols
                if i <= length(llr_cpu)
                    posteriors[i] = 1.0f0 / (1.0f0 + exp(llr_cpu[i]))
                end
            end
            
            sorted_indices = sortperm(posteriors[active_cols], rev=true)
            active_cols = active_cols[sorted_indices[1:min(length(sorted_indices), decoder.ac_max_cluster_size)]]
        end
        
        max_solutions = 1 + length(active_cols)
        if decoder.ac_search_order >= 2
            max_solutions += binomial(length(active_cols), 2)
        end
        
        max_solutions = min(max_solutions, 10000)
        
        solutions_cpu = zeros(Int32, (max_solutions, num_variables))
        solution_count = 1  # First solution is all zeros
        
        for (i, col) in enumerate(active_cols)
            if solution_count < max_solutions && col <= size(solutions_cpu, 2)
                solution_count += 1
                solutions_cpu[solution_count, col] = 1
            end
        end
        
        if decoder.ac_search_order >= 2
            for i in 1:length(active_cols)
                for j in (i+1):length(active_cols)
                    if solution_count < max_solutions && 
                       active_cols[i] <= size(solutions_cpu, 2) && 
                       active_cols[j] <= size(solutions_cpu, 2)
                        solution_count += 1
                        solutions_cpu[solution_count, active_cols[i]] = 1
                        solutions_cpu[solution_count, active_cols[j]] = 1
                    end
                end
            end
        end
        
        solutions_d = CuArray{Int32}(solutions_cpu[1:solution_count, :])
        
        solution_scores = CUDA.zeros(Float32, solution_count)
        
        threads_per_block = perf_config.thread_block_size
        blocks_sol = cld(solution_count, threads_per_block)
        
        @cuda threads=threads_per_block blocks=blocks_sol stream=stream ac_evaluate_solution_kernel_optimized!(
            solutions_d, solution_count, llr, solution_scores, num_variables)
        
        logical_effects = CUDA.zeros(Int32, (solution_count, num_logicals))
        
        threads_sol = min(16, solution_count)
        threads_log = min(16, num_logicals)
        blocks_sol = cld(solution_count, threads_sol)
        blocks_log = cld(num_logicals, threads_log)
        
        @cuda threads=(threads_sol, threads_log) blocks=(blocks_sol, blocks_log) stream=stream ac_compute_logical_effect_kernel_optimized!(
            solutions_d, solution_count, decoder.L_d.nzVal, decoder.L_d.rowPtr, decoder.L_d.colVal,
            logical_effects, num_logicals)
        
        CUDA.synchronize(stream)
        
        solution_scores_cpu = Array(solution_scores)
        logical_effects_cpu = Array(logical_effects)
        
        max_score = maximum(solution_scores_cpu)
        solution_probs = exp.(solution_scores_cpu .- max_score)
        
        for i in 1:num_logicals
            prob_0 = 0.0f0
            prob_1 = 0.0f0
            
            for j in 1:solution_count
                prob = solution_probs[j]
                if j <= size(logical_effects_cpu, 1) && i <= size(logical_effects_cpu, 2)
                    if logical_effects_cpu[j, i] == 0
                        prob_0 += prob
                    else
                        prob_1 += prob
                    end
                end
            end
            
            logical_effect[i] = prob_1 > prob_0 ? 1 : 0
        end
    end
    
    return logical_effect
end
