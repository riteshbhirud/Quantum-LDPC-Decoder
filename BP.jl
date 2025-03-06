using CUDA
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using DelimitedFiles

"""
    LDPCDecoder

A GPU-accelerated LDPC decoder implementing the normalized min-sum algorithm with flooding schedule.
TODO: optimize for quantum error correction applications.
"""
struct LDPCDecoder
    H::SparseMatrixCSC{Float32, Int32}  # Parity check matrix (CPU)
    H_d::CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}  # Parity check matrix (GPU)
    max_iterations::Int                   # Maximum number of iterations
    normalization_factor::Float32         # Normalization factor for min-sum
    check_indices::CuVector{Int32}        # Variable indices for each check node
    check_indices_ptr::CuVector{Int32}    # Pointers to variable indices for each check node
    variable_indices::CuVector{Int32}     # Check indices for each variable node
    variable_indices_ptr::CuVector{Int32} # Pointers to check indices for each variable node
    num_checks::Int                       # Number of check nodes
    num_variables::Int                    # Number of variable nodes
    
    function LDPCDecoder(H::SparseMatrixCSC{T}, max_iterations::Int=100, normalization_factor::Float32=0.75f0) where T
        H_f32 = convert(SparseMatrixCSC{Float32, Int32}, H)
        
        num_checks, num_variables = size(H_f32)
        
        check_indices_data = Vector{Vector{Int32}}(undef, num_checks)
        for i in 1:num_checks
            check_indices_data[i] = findall(!iszero, H_f32[i, :])
        end
        
        check_indices_flat = Int32[]
        check_indices_ptr = Int32[1]
        for indices in check_indices_data
            append!(check_indices_flat, indices)
            push!(check_indices_ptr, length(check_indices_flat) + 1)
        end
        
        variable_indices_data = Vector{Vector{Int32}}(undef, num_variables)
        for j in 1:num_variables
            variable_indices_data[j] = findall(!iszero, H_f32[:, j])
        end
        
        variable_indices_flat = Int32[]
        variable_indices_ptr = Int32[1]
        for indices in variable_indices_data
            append!(variable_indices_flat, indices)
            push!(variable_indices_ptr, length(variable_indices_flat) + 1)
        end
        
        # Transfer to GPU
        check_indices_gpu = CuVector{Int32}(check_indices_flat)
        check_indices_ptr_gpu = CuVector{Int32}(check_indices_ptr)
        variable_indices_gpu = CuVector{Int32}(variable_indices_flat)
        variable_indices_ptr_gpu = CuVector{Int32}(variable_indices_ptr)
        
        # Convert H to GPU CSR format for syndrome computation
        H_d = CUDA.CUSPARSE.CuSparseMatrixCSR(H_f32)
        
        new(H_f32, H_d, max_iterations, normalization_factor, 
            check_indices_gpu, check_indices_ptr_gpu, 
            variable_indices_gpu, variable_indices_ptr_gpu, 
            num_checks, num_variables)
    end
end

"""
    variable_to_check_kernel!(v2c_messages, llr, variable_indices, variable_indices_ptr, c2v_messages, num_variables, num_checks)

CUDA kernel for the variable-to-check message passing phase.
"""
function variable_to_check_kernel!(v2c_messages, llr, variable_indices, variable_indices_ptr, c2v_messages, num_variables)
    var_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if var_idx <= num_variables
        var_llr = llr[var_idx]
        
        start_idx = variable_indices_ptr[var_idx]
        end_idx = variable_indices_ptr[var_idx + 1] - 1
        
        for j in start_idx:end_idx
            check_idx = variable_indices[j]
            
            sum_msgs = var_llr
            for k in start_idx:end_idx
                if k != j
                    c_idx = variable_indices[k]
                    sum_msgs += c2v_messages[var_idx, c_idx]
                end
            end
            
            v2c_messages[var_idx, check_idx] = sum_msgs
        end
    end
    
    return nothing
end

"""
    check_to_variable_kernel!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, normalization_factor, num_checks)

CUDA kernel for the check-to-variable message passing phase using normalized min-sum.
"""
function check_to_variable_kernel!(c2v_messages, v2c_messages, check_indices, check_indices_ptr, normalization_factor, num_checks)
    check_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if check_idx <= num_checks
        # Get all variable nodes connected to this check
        start_idx = check_indices_ptr[check_idx]
        end_idx = check_indices_ptr[check_idx + 1] - 1
        
        for j in start_idx:end_idx
            var_idx = check_indices[j]
            
            min_val = Inf32
            sign_prod = 1.0f0
            
            for k in start_idx:end_idx
                other_var_idx = check_indices[k]
                if other_var_idx != var_idx
                    val = v2c_messages[other_var_idx, check_idx]
                    min_val = min(min_val, abs(val))
                    sign_prod *= sign(val)
                end
            end
            
            # Apply normalization factor (scaled min-sum)
            c2v_messages[var_idx, check_idx] = sign_prod * normalization_factor * min_val
        end
    end
    
    return nothing
end

"""
    compute_syndrome_kernel!(syndrome, hard_decision, H_indices, H_values, H_rows, H_cols)

CUDA kernel to compute the syndrome of the current hard decision.
"""
function compute_syndrome_kernel!(syndrome, hard_decision, H_values, H_row_ptr, H_col_ind)
    check_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if check_idx <= length(syndrome)
        result = 0
        
        for j in H_row_ptr[check_idx]:(H_row_ptr[check_idx+1]-1)
            col = H_col_ind[j]
            val = H_values[j]
            
            if val != 0 && hard_decision[col] == 1
                result = (result + 1) % 2
            end
        end
        
        syndrome[check_idx] = result
    end
    
    return nothing
end

"""
    update_beliefs_kernel!(beliefs, llr, c2v_messages, variable_indices, variable_indices_ptr, num_variables)

CUDA kernel to update belief values.
"""
function update_beliefs_kernel!(beliefs, llr, c2v_messages, variable_indices, variable_indices_ptr, num_variables)
    var_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if var_idx <= num_variables
        belief = llr[var_idx]
        
        start_idx = variable_indices_ptr[var_idx]
        end_idx = variable_indices_ptr[var_idx + 1] - 1
        
        for j in start_idx:end_idx
            check_idx = variable_indices[j]
            belief += c2v_messages[var_idx, check_idx]
        end
        
        beliefs[var_idx] = belief
    end
    
    return nothing
end

"""
    hard_decision_kernel!(hard_decisions, beliefs, num_variables)

CUDA kernel to make hard decisions based on beliefs.
"""
function hard_decision_kernel!(hard_decisions, beliefs, num_variables)
    var_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if var_idx <= num_variables
        hard_decisions[var_idx] = beliefs[var_idx] < 0 ? 1 : 0
    end
    
    return nothing
end

"""
    decode(decoder::LDPCDecoder, llr::Vector{Float32})

Decode the received message using the normalized min-sum algorithm with flooding schedule.

Parameters:
- decoder: The LDPC decoder
- llr: Log-likelihood ratios (LLRs) for each bit, where llr[i] = log(P(x_i=0) / P(x_i=1))

Returns:
- hard_decision: The decoded codeword
- success: Whether the decoding was successful (syndrome is zero)
- iterations: Number of iterations performed
"""
function decode(decoder::LDPCDecoder, llr::Vector{Float32})
    llr_gpu = CuVector{Float32}(llr)
    
    num_checks = decoder.num_checks
    num_variables = decoder.num_variables
    
    # Initialize messages
    v2c_messages = CUDA.zeros(Float32, (num_variables, num_checks))
    c2v_messages = CUDA.zeros(Float32, (num_variables, num_checks))
    
    # Initialize beliefs and hard decisions
    beliefs = CUDA.zeros(Float32, num_variables)
    hard_decisions = CUDA.zeros(Int32, num_variables)
    syndrome = CUDA.zeros(Int32, num_checks)
    
    H_values = decoder.H_d.nzVal
    H_row_ptr = decoder.H_d.rowPtr
    H_col_ind = decoder.H_d.colVal
    
    threads_per_block = 256
    blocks_var = cld(num_variables, threads_per_block)
    blocks_check = cld(num_checks, threads_per_block)
    
    # Main decoding loop
    for iteration in 1:decoder.max_iterations
        @cuda threads=threads_per_block blocks=blocks_var variable_to_check_kernel!(
            v2c_messages, llr_gpu, 
            decoder.variable_indices, decoder.variable_indices_ptr, 
            c2v_messages, num_variables
        )
        CUDA.synchronize()
        
        @cuda threads=threads_per_block blocks=blocks_check check_to_variable_kernel!(
            c2v_messages, v2c_messages, 
            decoder.check_indices, decoder.check_indices_ptr, 
            decoder.normalization_factor, num_checks
        )
        CUDA.synchronize()
        
        # Update beliefs
        @cuda threads=threads_per_block blocks=blocks_var update_beliefs_kernel!(
            beliefs, llr_gpu, c2v_messages, 
            decoder.variable_indices, decoder.variable_indices_ptr, 
            num_variables
        )
        CUDA.synchronize()
        
        # Make hard decisions
        @cuda threads=threads_per_block blocks=blocks_var hard_decision_kernel!(
            hard_decisions, beliefs, num_variables
        )
        CUDA.synchronize()
        
        # Compute syndrome
        @cuda threads=threads_per_block blocks=blocks_check compute_syndrome_kernel!(
            syndrome, hard_decisions, H_values, H_row_ptr, H_col_ind
        )
        CUDA.synchronize()
        
        # Check...
        syndrome_host = Array(syndrome)
        if all(syndrome_host .== 0)
            return Array(hard_decisions), true, iteration
        end
    end
    
    return Array(hard_decisions), false, decoder.max_iterations
end

"""
    create_random_ldpc(n, k, max_degree=6)

Create a random LDPC code with variable node degree and check node degree <= max_degree.

Parameters:
- n: Codeword length
- k: Message length (resulting in n-k check equations)
- max_degree: Maximum degree of variable and check nodes

Returns:
- H: Parity check matrix
"""
function create_random_ldpc(n, k, max_degree=6)
    m = n - k 
    
    density = min(max_degree / n, max_degree / m, 0.5)
    H = sprand(Bool, m, n, density)
    
    for i in 1:m
        row_weight = sum(H[i, :])
        if row_weight < 2
            zeros_idx = findall(x -> x == 0, H[i, :])
            num_to_add = 2 - row_weight
            add_idx = sample(zeros_idx, min(num_to_add, length(zeros_idx)), replace=false)
            H[i, add_idx] .= 1
        elseif row_weight > max_degree
            ones_idx = findall(x -> x == 1, H[i, :])
            num_to_remove = row_weight - max_degree
            remove_idx = sample(ones_idx, min(num_to_remove, length(ones_idx)), replace=false)
            H[i, remove_idx] .= 0
        end
    end
    
    for j in 1:n
        col_weight = sum(H[:, j])
        if col_weight < 2
            zeros_idx = findall(x -> x == 0, H[:, j])
            num_to_add = 2 - col_weight
            add_idx = sample(zeros_idx, min(num_to_add, length(zeros_idx)), replace=false)
            H[add_idx, j] .= 1
        elseif col_weight > max_degree
            ones_idx = findall(x -> x == 1, H[:, j])
            num_to_remove = col_weight - max_degree
            remove_idx = sample(ones_idx, min(num_to_remove, length(ones_idx)), replace=false)
            H[remove_idx, j] .= 0
        end
    end
    
    return H
end

"""
    simulate_bsc(codeword, p)

Simulate a binary symmetric channel (BSC) with error probability p.

Parameters:
- codeword: The codeword to transmit
- p: The error probability

Returns:
- received: The received vector
"""
function simulate_bsc(codeword, p)
    n = length(codeword)
    error_pattern = rand(n) .< p
    received = xor.(codeword, error_pattern)
    return received
end

"""
    llr_from_bsc(received, p)

Compute log-likelihood ratios (LLRs) for a binary symmetric channel (BSC).

Parameters:
- received: The received vector
- p: The error probability

Returns:
- llr: The LLRs
"""
function llr_from_bsc(received, p)
    llr = zeros(Float32, length(received))
    for i in 1:length(received)
        if received[i] == 0
            llr[i] = log((1-p) / p)
        else
            llr[i] = log(p / (1-p))
        end
    end
    return llr
end

"""
    Example usage for quantum error correction
"""
function example_qec()
    n = 1024 
    k = 512  
    p = 0.05 
    
    H = create_random_ldpc(n, k, 6)
    
    decoder = LDPCDecoder(H, 50, 0.8f0)
    
    # simple BSC model for demonstration
    
    codeword = zeros(Int, n)
    
    received = simulate_bsc(codeword, p)
    
    llr = llr_from_bsc(received, p)
    
    decoded, success, iterations = decode(decoder, llr)
    
    println("Decoded in $iterations iterations. Success: $success")
    error_rate = sum(decoded .!= codeword) / n
    println("Bit error rate: $error_rate")
    
    return decoded, success, iterations
end

"""
    performance_benchmark(n_values, p, num_trials)

Benchmark the decoder performance for different code sizes.

Parameters:
- n_values: List of codeword lengths to test
- p: Error probability
- num_trials: Number of trials for each configuration

Returns:
- results: Dictionary with performance metrics
"""
function performance_benchmark(n_values, p, num_trials)
    results = Dict()
    
    for n in n_values
        k = n ÷ 2  # Rate 1/2 code
        
        success_rate = 0.0
        avg_iterations = 0.0
        decoding_times = Float64[]
        
        H = create_random_ldpc(n, k, 6)
        decoder = LDPCDecoder(H, 50, 0.8f0)
        
        for _ in 1:num_trials
            codeword = zeros(Int, n)
            
            received = simulate_bsc(codeword, p)
            
            llr = llr_from_bsc(received, p)
            
            t_start = time()
            decoded, success, iterations = decode(decoder, llr)
            t_end = time()
            
            success_rate += success ? 1.0 : 0.0
            avg_iterations += iterations
            push!(decoding_times, t_end - t_start)
        end
        
        success_rate /= num_trials
        avg_iterations /= num_trials
        avg_time = mean(decoding_times)
        
        results[n] = (
            success_rate = success_rate,
            avg_iterations = avg_iterations,
            avg_time = avg_time,
        )
        
        println("n=$n: Success rate: $(success_rate), Avg iterations: $(avg_iterations), Avg time: $(avg_time) seconds")
    end
    
    return results
end

"""
    load_quantum_ldpc_code(filename)

Load a quantum LDPC code from a file.
For quantum stabilizer codes, the parity check matrix defines the stabilizers.

Parameters:
- filename: Path to the file containing the parity check matrix

Returns:
- H: Parity check matrix
"""
function load_quantum_ldpc_code(filename)
    # todo:This is a placeholder for loading real quantum LDPC codes
  
    
    H_data = readdlm(filename, Int)
    m, n = size(H_data)
    
    H = sparse(H_data)
    return H
end
using StatsBase: sample

# Main driver code that can be used for QEC applications
function main()
    println("="^50)
    println("RUNNING QEC EXAMPLE")
    println("="^50)
    println("Using a random LDPC code (1024 bits, rate 1/2) with simulated errors")
    println("This demonstrates application for quantum error correction")
    
    # Run the QEC example
    result, success, iterations = example_qec()
    
    # Print a separator
    println("\n")
    println("="^50)
    println("RUNNING PERFORMANCE BENCHMARK")
    println("="^50)
    
    n_values = [128, 256, 512] 
    p = 0.05
    num_trials = 5  
    
    println("Running benchmarks for code sizes: $n_values")
    println("Error probability: $p")
    println("Number of trials per code: $num_trials")
    println()
    
    results = performance_benchmark(n_values, p, num_trials)
    
    println("\nPerformance Summary:")
    println("-"^50)
    println("| Code Size | Success Rate | Avg Iterations | Avg Time (s) |")
    println("|" * "-"^10 * "|" * "-"^14 * "|" * "-"^16 * "|" * "-"^14 * "|")
    
    for n in n_values
        r = results[n]
        sr = round(r.success_rate * 100, digits=1)
        ai = round(r.avg_iterations, digits=1)
        at = round(r.avg_time, digits=4)
        println("| $n" * " "^(10-length("$n")) * 
                "| $sr%" * " "^(14-length("$sr%")) * 
                "| $ai" * " "^(16-length("$ai")) * 
                "| $at" * " "^(14-length("$at")) * "|")
    end
    println("-"^50)
    
    println("\nNOTE: For quantum error correction applications, you can:")
    println("1. Create your own parity check matrix for a specific QEC code")
    println("2. Adjust the normalization factor (currently 0.8) for better performance")
    println("3. Load your syndrome measurements from file")
    println("\nThis implementation is optimized for CUDA GPU acceleration")
end

    #main()
    using Random
    using DelimitedFiles
    using SparseArrays
    using Statistics
    using LinearAlgebra
    using Plots
    using StatsBase
    
    
    #  create a regular LDPC parity check matrix
    function parity_check_matrix(n::Int, wr::Int, wc::Int)
   
      ## wr = wc * (n / n-k)
      @assert n % wr == 0
      n_equations = (n * wc) ÷ wr
      block_size = n_equations ÷ wc
      block = zeros(Bool, block_size, n)
      for i in 1:block_size
        for j in ((i-1)*wr + 1):((i)*wr)
          block[i,j] = 1
        end
      end
      H = block
      for i in 1:wc - 1
        H = [H; block[:, shuffle(1:end)]]
      end
      return BitArray(H)
    end
    
    function save_pcm(H, file_path)
      writedlm(file_path, Int.(H))
    end
    
    function load_pcm(file_path)
      H = readdlm(file_path)
      return Int.(H)
    end
    
    function test_regular_codes()
        n_values = [128, 256, 512]
        wr_values = [4, 6, 8]
        wc_values = [3, 4]
        
        p_values = [0.01, 0.05, 0.1]
        
        num_trials = 10
        
        results = Dict()
        
        for n in n_values
            for wr in wr_values
                for wc in wc_values
                    if n % wr != 0 || (n * wc) % wr != 0
                        println("Skipping invalid configuration: n=$n, wr=$wr, wc=$wc")
                        continue
                    end
                    
                    println("Generating regular LDPC code: n=$n, wr=$wr, wc=$wc")
                    H = parity_check_matrix(n, wr, wc)
                    
                    H_sparse = sparse(H)
                    
                    m = size(H, 1)
                    
                    rate = (n - m) / n
                    
                    println("  Code rate: $rate")
                    println("  Matrix size: $(size(H))")
                    
                    for p in p_values
                        config_key = "n$(n)_wr$(wr)_wc$(wc)_p$(p)"
                        results[config_key] = Dict(
                            "success_rate" => 0.0,
                            "avg_iterations" => 0.0,
                            "avg_time" => 0.0,
                            "bit_error_rate" => 0.0,
                            "n" => n,
                            "wr" => wr,
                            "wc" => wc,
                            "p" => p,
                            "rate" => rate
                        )
                        
                        println("  Testing with error probability: $p")
                        
                        decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
                        
                        success_count = 0
                        total_iterations = 0
                        total_time = 0.0
                        total_bit_errors = 0
                        
                        for trial in 1:num_trials
                            # (all zeros for simplicity)
                            codeword = zeros(Int, n)
                            
                            received = simulate_bsc(codeword, p)
                            
                            initial_errors = sum(received)
                            
                            llr = llr_from_bsc(received, p)
                            
                            t_start = time()
                            decoded, success, iterations = decode(decoder, llr)
                            t_end = time()
                            
                            success_count += success ? 1 : 0
                            total_iterations += iterations
                            total_time += (t_end - t_start)
                            total_bit_errors += sum(decoded .!= codeword)
                            
                            if trial % 2 == 0
                                print(".") 
                            end
                        end
                        println()  
                        
                        success_rate = success_count / num_trials
                        avg_iterations = total_iterations / num_trials
                        avg_time = total_time / num_trials
                        bit_error_rate = total_bit_errors / (n * num_trials)
                        
                        # Store results
                        results[config_key]["success_rate"] = success_rate
                        results[config_key]["avg_iterations"] = avg_iterations
                        results[config_key]["avg_time"] = avg_time
                        results[config_key]["bit_error_rate"] = bit_error_rate
                        
                        println("    Success rate: $(success_rate * 100)%")
                        println("    Avg iterations: $avg_iterations")
                        println("    Avg time: $(avg_time * 1000) ms")
                        println("    Bit error rate: $(bit_error_rate * 100)%")
                    end
                end
            end
        end
        
        return results
    end
    
    function test_normalization_factors()
        n = 256
        wr = 8
        wc = 4
        p = 0.05
        
        num_trials = 20
        
        factors = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        
        println("Generating regular LDPC code: n=$n, wr=$wr, wc=$wc")
        H = parity_check_matrix(n, wr, wc)
        H_sparse = sparse(H)
        
        results = Dict()
        
        for factor in factors
            println("Testing normalization factor: $factor")
            
            decoder = LDPCDecoder(H_sparse, 50, convert(Float32, factor))
            
            success_count = 0
            total_iterations = 0
            
            for trial in 1:num_trials
                codeword = zeros(Int, n)
                
                received = simulate_bsc(codeword, p)
                
                llr = llr_from_bsc(received, p)
                
                decoded, success, iterations = decode(decoder, llr)
                
                success_count += success ? 1 : 0
                total_iterations += iterations
                
                if trial % 4 == 0
                    print(".")  # Show progress
                end
            end
            println() 
            
            success_rate = success_count / num_trials
            avg_iterations = total_iterations / num_trials
            
            results[factor] = (
                success_rate = success_rate,
                avg_iterations = avg_iterations
            )
            
            println("  Success rate: $(success_rate * 100)%")
            println("  Avg iterations: $avg_iterations")
        end
        
        return results
    end
    
    function test_quantum_error_model(n, wr, wc, p_x, p_z, p_y)
        println("Generating CSS-like quantum LDPC code: n=$n, wr=$wr, wc=$wc")
        H_x = parity_check_matrix(n, wr, wc)  
        H_z = parity_check_matrix(n, wr, wc)  
        
 
        H_x_sparse = sparse(H_x)
        H_z_sparse = sparse(H_z)
        
        x_decoder = LDPCDecoder(H_x_sparse, 50, 0.8f0)
        z_decoder = LDPCDecoder(H_z_sparse, 50, 0.8f0)
        
        num_trials = 50
        
        x_success = 0
        z_success = 0
        
        for trial in 1:num_trials
           
            x_errors = rand(n) .< (p_x + p_y) 
            z_errors = rand(n) .< (p_z + p_y)  
            
            x_syndrome = (H_z * x_errors) .% 2  
            z_syndrome = (H_x * z_errors) .% 2 
            
            # Convert syndromes to LLRs (simplified model)
     
            x_llr = zeros(Float32, n)
            z_llr = zeros(Float32, n)
            
            # Set high certainty for syndrome bit = 1
            for i in 1:length(x_syndrome)
                if x_syndrome[i] == 1
                 
                    connected_qubits = findall(!iszero, H_z[i, :])
                    for q in connected_qubits
                        x_llr[q] -= 2.0
                    end
                end
            end
            
            for i in 1:length(z_syndrome)
                if z_syndrome[i] == 1
                    connected_qubits = findall(!iszero, H_x[i, :])
                    for q in connected_qubits
                        z_llr[q] -= 2.0
                    end
                end
            end
            
            x_decoded, x_success_trial, _ = decode(x_decoder, x_llr)
            z_decoded, z_success_trial, _ = decode(z_decoder, z_llr)
            
            x_success += x_success_trial ? 1 : 0
            z_success += z_success_trial ? 1 : 0
            
            if trial % 5 == 0
                print(".")
            end
        end
        println()
        
        x_success_rate = x_success / num_trials
        z_success_rate = z_success / num_trials
        
        println("X error success rate: $(x_success_rate * 100)%")
        println("Z error success rate: $(z_success_rate * 100)%")
        println("Overall success rate: $((x_success_rate * z_success_rate) * 100)%")
        
        return (x_success_rate, z_success_rate)
    end
    
    function run_all_tests()
        println("="^50)
        println("TESTING REGULAR LDPC CODES")
        println("="^50)
        regular_results = test_regular_codes()
        
        println("\n")
        println("="^50)
        println("TESTING NORMALIZATION FACTORS")
        println("="^50)
        norm_results = test_normalization_factors()
        
        println("\n")
        println("="^50)
        println("TESTING QUANTUM ERROR MODEL")
        println("="^50)
       
        qec_results = test_quantum_error_model(128, 4, 3, 0.01, 0.01, 0.001)
        
        return (regular_results, norm_results, qec_results)
    end
    
   
       using SparseArrays
       using LinearAlgebra
       using Random
       using Statistics
       using DelimitedFiles
       using StatsBase: sample
       
       """
           LDPCDecoderCPU
       
       A CPU implementation of LDPC decoder using the normalized min-sum algorithm 
       with flooding schedule. This is designed as a benchmark comparison against 
       the GPU-accelerated version.
       """
       struct LDPCDecoderCPU
           H::SparseMatrixCSC{Float32, Int32}  # Parity check matrix
           max_iterations::Int                   # Maximum number of iterations
           normalization_factor::Float32         # Normalization factor for min-sum
           check_indices::Vector{Vector{Int32}}  # Indices of variable nodes connected to each check node
           variable_indices::Vector{Vector{Int32}}  # Indices of check nodes connected to each variable node
           num_checks::Int                       # Number of check nodes
           num_variables::Int                    # Number of variable nodes
           
           function LDPCDecoderCPU(H::SparseMatrixCSC{T}, max_iterations::Int=100, normalization_factor::Float32=0.75f0) where T
               H_f32 = convert(SparseMatrixCSC{Float32, Int32}, H)
               
               num_checks, num_variables = size(H_f32)
               
               check_indices = Vector{Vector{Int32}}(undef, num_checks)
               for i in 1:num_checks
                   check_indices[i] = findall(!iszero, H_f32[i, :])
               end
               
               variable_indices = Vector{Vector{Int32}}(undef, num_variables)
               for j in 1:num_variables
                   variable_indices[j] = findall(!iszero, H_f32[:, j])
               end
               
               new(H_f32, max_iterations, normalization_factor, check_indices, variable_indices, num_checks, num_variables)
           end
       end
       
       """
           variable_to_check_messages(decoder, llr, c2v_messages)
       
       Compute the variable-to-check messages for the min-sum algorithm.
       
       Parameters:
       - decoder: The LDPC decoder
       - llr: Log-likelihood ratios for each bit
       - c2v_messages: Check-to-variable messages from previous iteration
       
       Returns:
       - v2c_messages: Variable-to-check messages
       """
       function variable_to_check_messages(decoder::LDPCDecoderCPU, llr::Vector{Float32}, c2v_messages::Matrix{Float32})
           num_variables = decoder.num_variables
           num_checks = decoder.num_checks
           
           v2c_messages = zeros(Float32, (num_variables, num_checks))
           
           for var_idx in 1:num_variables
               var_llr = llr[var_idx]
               
               connected_checks = decoder.variable_indices[var_idx]
               
               for check_idx in connected_checks
                   # Sum of all incoming check messages except current one
                   sum_msgs = var_llr
                   for other_check_idx in connected_checks
                       if other_check_idx != check_idx
                           sum_msgs += c2v_messages[var_idx, other_check_idx]
                       end
                   end
                   
                   v2c_messages[var_idx, check_idx] = sum_msgs
               end
           end
           
           return v2c_messages
       end
       
       """
           check_to_variable_messages(decoder, v2c_messages)
       
       Compute the check-to-variable messages using the normalized min-sum algorithm.
       
       Parameters:
       - decoder: The LDPC decoder
       - v2c_messages: Variable-to-check messages from current iteration
       
       Returns:
       - c2v_messages: Check-to-variable messages
       """
       function check_to_variable_messages(decoder::LDPCDecoderCPU, v2c_messages::Matrix{Float32})
           num_variables = decoder.num_variables
           num_checks = decoder.num_checks
           
           c2v_messages = zeros(Float32, (num_variables, num_checks))
           
           for check_idx in 1:num_checks
               connected_vars = decoder.check_indices[check_idx]
               
               for var_idx in connected_vars
                   min_val = Inf32
                   sign_prod = 1.0f0
                   
                   for other_var_idx in connected_vars
                       if other_var_idx != var_idx
                           val = v2c_messages[other_var_idx, check_idx]
                           min_val = min(min_val, abs(val))
                           sign_prod *= sign(val)
                       end
                   end
                   
                   # Apply normalization factor (scaled min-sum)
                   c2v_messages[var_idx, check_idx] = sign_prod * decoder.normalization_factor * min_val
               end
           end
           
           return c2v_messages
       end
       
       """
           update_beliefs(decoder, llr, c2v_messages)
       
       Update belief values for each variable node.
       
       Parameters:
       - decoder: The LDPC decoder
       - llr: Log-likelihood ratios for each bit
       - c2v_messages: Check-to-variable messages from current iteration
       
       Returns:
       - beliefs: Updated belief values
       """
       function update_beliefs(decoder::LDPCDecoderCPU, llr::Vector{Float32}, c2v_messages::Matrix{Float32})
           num_variables = decoder.num_variables
           
           beliefs = zeros(Float32, num_variables)
           
           for var_idx in 1:num_variables
               belief = llr[var_idx]
               
               connected_checks = decoder.variable_indices[var_idx]
               for check_idx in connected_checks
                   belief += c2v_messages[var_idx, check_idx]
               end
               
               beliefs[var_idx] = belief
           end
           
           return beliefs
       end
       
       """
           compute_syndrome(decoder, hard_decision)
       
       Compute the syndrome of the current hard decision.
       
       Parameters:
       - decoder: The LDPC decoder
       - hard_decision: Current hard decision for each bit
       
       Returns:
       - syndrome: Syndrome vector (0 indicates satisfied check equation)
       """
       function compute_syndrome(decoder::LDPCDecoderCPU, hard_decision::Vector{Int})
           H = decoder.H
           syndrome = zeros(Int, decoder.num_checks)
           
           syndrome_float = H * hard_decision
           syndrome = Int.(syndrome_float .% 2)
           
           return syndrome
       end
       
       """
           decode(decoder, llr)
       
       Decode the received message using the normalized min-sum algorithm with flooding schedule.
       
       Parameters:
       - decoder: The LDPC decoder
       - llr: Log-likelihood ratios (LLRs) for each bit, where llr[i] = log(P(x_i=0) / P(x_i=1))
       
       Returns:
       - hard_decision: The decoded codeword
       - success: Whether the decoding was successful (syndrome is zero)
       - iterations: Number of iterations performed
       """
       function decode(decoder::LDPCDecoderCPU, llr::Vector{Float32})
           num_checks = decoder.num_checks
           num_variables = decoder.num_variables
           
           v2c_messages = zeros(Float32, (num_variables, num_checks))
           c2v_messages = zeros(Float32, (num_variables, num_checks))
           
           for iteration in 1:decoder.max_iterations
               v2c_messages = variable_to_check_messages(decoder, llr, c2v_messages)
               
               c2v_messages = check_to_variable_messages(decoder, v2c_messages)
               
               beliefs = update_beliefs(decoder, llr, c2v_messages)
               
               hard_decisions = Int.(beliefs .< 0)
               
               syndrome = compute_syndrome(decoder, hard_decisions)
               
               if all(syndrome .== 0)
                   return hard_decisions, true, iteration
               end
           end
           
           hard_decisions = Int.(update_beliefs(decoder, llr, c2v_messages) .< 0)
           return hard_decisions, false, decoder.max_iterations
       end
       
       """
           create_random_ldpc(n, k, max_degree=6)
       
       Create a random LDPC code with variable node degree and check node degree <= max_degree.
       
       Parameters:
       - n: Codeword length
       - k: Message length (resulting in n-k check equations)
       - max_degree: Maximum degree of variable and check nodes
       
       Returns:
       - H: Parity check matrix
       """
       function create_random_ldpc(n, k, max_degree=6)
           m = n - k  
           
           density = min(max_degree / n, max_degree / m, 0.5)
           H = sprand(Bool, m, n, density)
           
           for i in 1:m
               row_weight = sum(H[i, :])
               if row_weight < 2
                   zeros_idx = findall(x -> x == 0, H[i, :])
                   num_to_add = 2 - row_weight
                   add_idx = sample(zeros_idx, min(num_to_add, length(zeros_idx)), replace=false)
                   H[i, add_idx] .= 1
               elseif row_weight > max_degree
                   ones_idx = findall(x -> x == 1, H[i, :])
                   num_to_remove = row_weight - max_degree
                   remove_idx = sample(ones_idx, min(num_to_remove, length(ones_idx)), replace=false)
                   H[i, remove_idx] .= 0
               end
           end
           
           for j in 1:n
               col_weight = sum(H[:, j])
               if col_weight < 2
                   zeros_idx = findall(x -> x == 0, H[:, j])
                   num_to_add = 2 - col_weight
                   add_idx = sample(zeros_idx, min(num_to_add, length(zeros_idx)), replace=false)
                   H[add_idx, j] .= 1
               elseif col_weight > max_degree
                   ones_idx = findall(x -> x == 1, H[:, j])
                   num_to_remove = col_weight - max_degree
                   remove_idx = sample(ones_idx, min(num_to_remove, length(ones_idx)), replace=false)
                   H[remove_idx, j] .= 0
               end
           end
           
           return H
       end
       
       """
           simulate_bsc(codeword, p)
       
       Simulate a binary symmetric channel (BSC) with error probability p.
       
       Parameters:
       - codeword: The codeword to transmit
       - p: The error probability
       
       Returns:
       - received: The received vector
       """
       function simulate_bsc(codeword, p)
           n = length(codeword)
           error_pattern = rand(n) .< p
           received = xor.(codeword, error_pattern)
           return received
       end
       
       """
           llr_from_bsc(received, p)
       
       Compute log-likelihood ratios (LLRs) for a binary symmetric channel (BSC).
       
       Parameters:
       - received: The received vector
       - p: The error probability
       
       Returns:
       - llr: The LLRs
       """
       function llr_from_bsc(received, p)
           llr = zeros(Float32, length(received))
           for i in 1:length(received)
               if received[i] == 0
                   llr[i] = log((1-p) / p)
               else
                   llr[i] = log(p / (1-p))
               end
           end
           return llr
       end
       
       """
           parity_check_matrix(n, wr, wc)
       
       Create a regular LDPC parity check matrix with variable node degree wc and check node degree wr.
       
       Parameters:
       - n: Codeword length
       - wr: Row weight (check node degree)
       - wc: Column weight (variable node degree)
       
       Returns:
       - H: Parity check matrix
       """
       function parity_check_matrix(n::Int, wr::Int, wc::Int)
           # For a regular LDPC matrix
           ## wr = wc * (n / n-k)
           @assert n % wr == 0
           n_equations = (n * wc) ÷ wr
           block_size = n_equations ÷ wc
           block = zeros(Bool, block_size, n)
           for i in 1:block_size
               for j in ((i-1)*wr + 1):((i)*wr)
                   block[i,j] = 1
               end
           end
           H = block
           for i in 1:wc - 1
               H = [H; block[:, shuffle(1:end)]]
           end
           return BitArray(H)
       end
       
       """
           save_pcm(H, file_path)
       
       Save a parity check matrix to a file.
       
       Parameters:
       - H: Parity check matrix
       - file_path: Path to save the matrix
       """
       function save_pcm(H, file_path)
           writedlm(file_path, Int.(H))
       end
       
       """
           load_pcm(file_path)
       
       Load a parity check matrix from a file.
       
       Parameters:
       - file_path: Path to the file containing the matrix
       
       Returns:
       - H: Parity check matrix
       """
       function load_pcm(file_path)
           H = readdlm(file_path)
           return BitArray(H)
       end

       using Random
       using SparseArrays
       using DelimitedFiles
       using Statistics
       using Printf
       using StatsBase: sample
  
       
       """
           benchmark_cpu_gpu(n_values, wr, wc, p, num_trials)
       
       Benchmark CPU vs GPU LDPC decoders across different code sizes.
       
       Parameters:
       - n_values: Array of code sizes to test
       - wr: Row weight
       - wc: Column weight  
       - p: Error probability
       - num_trials: Number of trials for each configuration
       
       Returns:
       - results: Dictionary with benchmark results
       """
       function benchmark_cpu_gpu(n_values, wr, wc, p, num_trials)
           println("="^60)
           println("BENCHMARK: CPU vs GPU LDPC DECODER")
           println("="^60)
           println("Row weight (wr): $wr, Column weight (wc): $wc")
           println("Error probability: $p")
           println("Number of trials per configuration: $num_trials")
           println("-"^60)
           
           @printf("| %10s | %10s | %15s | %15s | %10s |\n", 
                  "Code Size", "Code Rate", "CPU Time (ms)", "GPU Time (ms)", "Speedup")
           println("|" * "-"^12 * "|" * "-"^12 * "|" * "-"^17 * "|" * "-"^17 * "|" * "-"^12 * "|")
           
           results = Dict()
           
           for n in n_values
               # ...skip case
               if n % wr != 0 || (n * wc) % wr != 0
                   @printf("| %10d | %10s | %15s | %15s | %10s |\n", 
                          n, "INVALID", "-", "-", "-")
                   continue
               end
               
               H = parity_check_matrix(n, wr, wc)
               H_sparse = sparse(Float32.(H))
               
               m = size(H, 1)
               rate = (n - m) / n
               
               cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
               gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
               
               cpu_times = Float64[]
               gpu_times = Float64[]
               cpu_success = 0
               gpu_success = 0
               
               for trial in 1:num_trials
                   codeword = zeros(Int, n)
                   
                   received = simulate_bsc(codeword, p)
                   llr = llr_from_bsc(received, p)
                   
                   t_start = time()
                   cpu_decoded, cpu_success_trial, cpu_iterations = decode(cpu_decoder, llr)
                   t_end = time()
                   cpu_time = (t_end - t_start) * 1000  
                   push!(cpu_times, cpu_time)
                   cpu_success += cpu_success_trial ? 1 : 0
                   
                   # GPU timing
                   t_start = time()
                   gpu_decoded, gpu_success_trial, gpu_iterations = decode(gpu_decoder, llr)
                   t_end = time()
                   gpu_time = (t_end - t_start) * 1000  
                   push!(gpu_times, gpu_time)
                   gpu_success += gpu_success_trial ? 1 : 0
               end
               
               avg_cpu_time = mean(cpu_times)
               avg_gpu_time = mean(gpu_times)
               
               speedup = avg_cpu_time / avg_gpu_time
               
               results[n] = (
                   rate = rate,
                   cpu_time = avg_cpu_time,
                   gpu_time = avg_gpu_time,
                   speedup = speedup,
                   cpu_success_rate = cpu_success / num_trials,
                   gpu_success_rate = gpu_success / num_trials
               )
               
               @printf("| %10d | %10.3f | %15.3f | %15.3f | %10.2fx |\n", 
                      n, rate, avg_cpu_time, avg_gpu_time, speedup)
           end
           
           return results
       end
       
       """
           benchmark_decoding_speed(n, wr, wc, p, iterations, warming=true)
       
       Benchmark pure decoding speed without setup overhead.
       
       Parameters:
       - n: Code size
       - wr: Row weight
       - wc: Column weight
       - p: Error probability
       - iterations: Number of iterations to run
       - warming: Whether to perform warming runs before timing
       
       Returns:
       - cpu_time: Average CPU time per decode
       - gpu_time: Average GPU time per decode
       - speedup: GPU speedup factor
       """
       function benchmark_decoding_speed(n, wr, wc, p, iterations, warming=true)
           println("\n")
           println("="^60)
           println("DECODING SPEED BENCHMARK (n=$n, wr=$wr, wc=$wc)")
           println("="^60)
           println("Measuring pure decoding speed over $iterations iterations")
           println("Error probability: $p")
           
           H = parity_check_matrix(n, wr, wc)
           H_sparse = sparse(Float32.(H))
           
           cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
           gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
           
           codeword = zeros(Int, n)
           received = simulate_bsc(codeword, p)
           llr = llr_from_bsc(received, p)
           
           if warming
               println("Performing warming runs...")
               for _ in 1:5
                   decode(cpu_decoder, llr)
                   decode(gpu_decoder, llr)
               end
           end
           
           println("Running CPU benchmark...")
           cpu_start = time()
           for _ in 1:iterations
               decode(cpu_decoder, llr)
           end
           cpu_end = time()
           cpu_total = cpu_end - cpu_start
           cpu_per_decode = (cpu_total / iterations) * 1000  # ms
           
           println("Running GPU benchmark...")
           gpu_start = time()
           for _ in 1:iterations
               decode(gpu_decoder, llr)
           end
           gpu_end = time()
           gpu_total = gpu_end - gpu_start
           gpu_per_decode = (gpu_total / iterations) * 1000  # ms
           
           speedup = cpu_per_decode / gpu_per_decode
           
           println("-"^60)
           println("CPU time per decode: $(@sprintf("%.3f", cpu_per_decode)) ms")
           println("GPU time per decode: $(@sprintf("%.3f", gpu_per_decode)) ms")
           println("GPU speedup: $(@sprintf("%.2fx", speedup))")
           
           return cpu_per_decode, gpu_per_decode, speedup
       end
       
       """
           benchmark_scaling()
       
       Benchmark how CPU and GPU implementations scale with code size.
       
       Returns:
       - scaling_results: Dictionary with scaling benchmark results
       """
       function benchmark_scaling()
           println("\n")
           println("="^60)
           println("SCALING BENCHMARK: CPU vs GPU")
           println("="^60)
           println("Testing how performance scales with code size")
           
           n_values = [128, 256, 512, 1024, 2048, 4096, 8192]
           wr = 8
           wc = 4
           p = 0.02  
           num_trials = 5
           
           results = benchmark_cpu_gpu(n_values, wr, wc, p, num_trials)
           
           n_valid = [n for n in n_values if haskey(results, n)]
           speedups = [results[n].speedup for n in n_valid]
           
           println("\nScaling summary:")
           println("- Minimum speedup: $(@sprintf("%.2fx", minimum(speedups)))")
           println("- Maximum speedup: $(@sprintf("%.2fx", maximum(speedups)))")
           println("- Average speedup: $(@sprintf("%.2fx", mean(speedups)))")
           
           if length(speedups) > 1
               if speedups[end] > speedups[1]
                   println("- GPU advantage increases with code size")
               else
                   println("- GPU advantage decreases with code size")
               end
           end
           
           return results
       end
       
       """
           benchmark_error_rates()
       
       Benchmark CPU vs GPU implementations across different error rates.
       
       Returns:
       - error_rate_results: Dictionary with error rate benchmark results
       """
       function benchmark_error_rates()
           println("\n")
           println("="^60)
           println("ERROR RATE BENCHMARK: CPU vs GPU")
           println("="^60)
           println("Testing performance across different error rates")
           
           n = 1024
           wr = 8
           wc = 4
           p_values = [0.01, 0.02, 0.05, 0.08, 0.1]
           num_trials = 10
           
           H = parity_check_matrix(n, wr, wc)
           H_sparse = sparse(Float32.(H))
           
           cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
           gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
           
           println("\n| Error Rate | CPU Time (ms) | GPU Time (ms) | Speedup | CPU Success | GPU Success |")
           println("|" * "-"^12 * "|" * "-"^14 * "|" * "-"^14 * "|" * "-"^9 * "|" * "-"^13 * "|" * "-"^13 * "|")
           
           results = Dict()
           
           for p in p_values
               cpu_times = Float64[]
               gpu_times = Float64[]
               cpu_success = 0
               gpu_success = 0
               cpu_iterations = 0
               gpu_iterations = 0
               
               for trial in 1:num_trials
                   codeword = zeros(Int, n)
                   
                   received = simulate_bsc(codeword, p)
                   llr = llr_from_bsc(received, p)
                   
                   t_start = time()
                   cpu_decoded, cpu_success_trial, cpu_iters = decode(cpu_decoder, llr)
                   t_end = time()
                   cpu_time = (t_end - t_start) * 1000  
                   push!(cpu_times, cpu_time)
                   cpu_success += cpu_success_trial ? 1 : 0
                   cpu_iterations += cpu_iters
                   
                   # GPU timing
                   t_start = time()
                   gpu_decoded, gpu_success_trial, gpu_iters = decode(gpu_decoder, llr)
                   t_end = time()
                   gpu_time = (t_end - t_start) * 1000  
                   push!(gpu_times, gpu_time)
                   gpu_success += gpu_success_trial ? 1 : 0
                   gpu_iterations += gpu_iters
               end
               
               avg_cpu_time = mean(cpu_times)
               avg_gpu_time = mean(gpu_times)
               cpu_success_rate = cpu_success / num_trials
               gpu_success_rate = gpu_success / num_trials
               avg_cpu_iterations = cpu_iterations / num_trials
               avg_gpu_iterations = gpu_iterations / num_trials
               
               speedup = avg_cpu_time / avg_gpu_time
               
               results[p] = (
                   cpu_time = avg_cpu_time,
                   gpu_time = avg_gpu_time,
                   speedup = speedup,
                   cpu_success_rate = cpu_success_rate,
                   gpu_success_rate = gpu_success_rate,
                   cpu_iterations = avg_cpu_iterations,
                   gpu_iterations = avg_gpu_iterations
               )
               
               @printf("| %10.3f | %12.3f | %12.3f | %7.2fx | %11.1f%% | %11.1f%% |\n", 
                      p, avg_cpu_time, avg_gpu_time, speedup, cpu_success_rate*100, gpu_success_rate*100)
           end
           
           return results
       end
       
       """
           run_all_benchmarks()
       
       Run all benchmark tests and return the combined results.
       """
       function run_all_benchmarks()
           println("RUNNING COMPREHENSIVE CPU vs GPU BENCHMARKS")
           println("This will test various aspects of decoder performance")
           
           println("\n[1/4] Testing multiple code sizes...")
           size_results = benchmark_cpu_gpu([128, 256, 512, 1024, 2048], 8, 4, 0.03, 10)
           
           println("\n[2/4] Testing pure decoding speed...")
           speed_results = benchmark_decoding_speed(1024, 8, 4, 0.03, 100)
           
           println("\n[3/4] Testing scaling with code size...")
           scaling_results = benchmark_scaling()
           
           println("\n[4/4] Testing performance across error rates...")
           error_rate_results = benchmark_error_rates()
           
           return (
               size_results = size_results,
               speed_results = speed_results,
               scaling_results = scaling_results,
               error_rate_results = error_rate_results
           )
       end
       #run_all_benchmarks()
       using Random
       using SparseArrays
       using DelimitedFiles
       using Statistics
       using Printf
       using StatsBase: sample
       using BenchmarkTools
       using CUDA
       

       
       """
           benchmark_decoding_only(n_values, wr, wc, p, num_trials=10)
       
       Benchmark only the decoding part of CPU and GPU LDPC decoders.
       This excludes initialization time, memory transfers, and other setup overhead.
       
       Parameters:
       - n_values: Array of code sizes to test
       - wr: Row weight
       - wc: Column weight  
       - p: Error probability
       - num_trials: Number of trials for each configuration
       
       Returns:
       - results: Dictionary with benchmark results
       """
       function benchmark_decoding_only(n_values, wr, wc, p, num_trials=10)
           println("="^60)
           println("DECODING-ONLY BENCHMARK: CPU vs GPU")
           println("="^60)
           println("Row weight (wr): $wr, Column weight (wc): $wc")
           println("Error probability: $p")
           println("Number of trials per configuration: $num_trials")
           println("-"^60)
           
           @printf("| %10s | %10s | %15s | %15s | %10s |\n", 
                  "Code Size", "Code Rate", "CPU Time (ms)", "GPU Time (ms)", "Speedup")
           println("|" * "-"^12 * "|" * "-"^12 * "|" * "-"^17 * "|" * "-"^17 * "|" * "-"^12 * "|")
           
           results = Dict()
           
           for n in n_values
               if n % wr != 0 || (n * wc) % wr != 0
                   @printf("| %10d | %10s | %15s | %15s | %10s |\n", 
                          n, "INVALID", "-", "-", "-")
                   continue
               end
               
               H = parity_check_matrix(n, wr, wc)
               H_sparse = sparse(Float32.(H))
               
               m = size(H, 1)
               rate = (n - m) / n
               
               cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
               gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
               
               cpu_times = Float64[]
               gpu_times = Float64[]
               
               for trial in 1:num_trials
                   codeword = zeros(Int, n)
                   received = simulate_bsc(codeword, p)
                   llr = llr_from_bsc(received, p)
                   
                   # ----------------
                   # CPU TIMING - ONLY THE DECODE FUNCTION
                   # ----------------
            
                   GC.gc() 
                   
                   t_start = time()
                   decode(cpu_decoder, llr)
                   t_end = time()
                   cpu_time_ms = (t_end - t_start) * 1000  
                   push!(cpu_times, cpu_time_ms)
                   
                   # ----------------
                   # GPU TIMING - ONLY THE DECODE FUNCTION
                   # ----------------
                   dummy = decode(gpu_decoder, llr)  
                   
                   t_start = time()
                   decode(gpu_decoder, llr)
                   CUDA.synchronize()
                   t_end = time()
                   gpu_time_ms = (t_end - t_start) * 1000  
                   push!(gpu_times, gpu_time_ms)
               end
               

               sort!(cpu_times)
               sort!(gpu_times)
               keep_count = max(1, round(Int, num_trials * 0.75))
               avg_cpu_time = mean(cpu_times[1:keep_count])
               avg_gpu_time = mean(gpu_times[1:keep_count])
               
               speedup = avg_cpu_time / avg_gpu_time
               
               results[n] = (
                   rate = rate,
                   cpu_time = avg_cpu_time,
                   gpu_time = avg_gpu_time,
                   speedup = speedup
               )
               
               @printf("| %10d | %10.3f | %15.3f | %15.3f | %10.2fx |\n", 
                      n, rate, avg_cpu_time, avg_gpu_time, speedup)
           end
           
           return results
       end
       
       """
           benchmark_fixed_iterations(n_values, wr, wc, p, fixed_iterations=20)
       
       Benchmark CPU and GPU with a fixed number of iterations to ensure fair comparison.
       This ensures both implementations run the same number of iterations regardless of
       early termination conditions.
       
       Parameters:
       - n_values: Array of code sizes to test
       - wr: Row weight
       - wc: Column weight  
       - p: Error probability
       - fixed_iterations: Number of iterations to force for both decoders
       
       Returns:
       - results: Dictionary with benchmark results
       """
       function benchmark_fixed_iterations(n_values, wr, wc, p, fixed_iterations=20)
           println("\n")
           println("="^60)
           println("FIXED ITERATIONS BENCHMARK: CPU vs GPU")
           println("="^60)
           println("Running exactly $fixed_iterations iterations for both decoders")
           println("Row weight (wr): $wr, Column weight (wc): $wc")
           println("Error probability: $p")
           println("-"^60)
           
           @printf("| %10s | %10s | %15s | %15s | %10s |\n", 
                  "Code Size", "Code Rate", "CPU Time (ms)", "GPU Time (ms)", "Speedup")
           println("|" * "-"^12 * "|" * "-"^12 * "|" * "-"^17 * "|" * "-"^17 * "|" * "-"^12 * "|")
           
           results = Dict()
           
           for n in n_values
               if n % wr != 0 || (n * wc) % wr != 0
                   continue
               end
               
               H = parity_check_matrix(n, wr, wc)
               H_sparse = sparse(Float32.(H))
               
               m = size(H, 1)
               rate = (n - m) / n
               
               cpu_decoder = LDPCDecoderCPU(H_sparse, fixed_iterations, 0.8f0)
               gpu_decoder = LDPCDecoder(H_sparse, fixed_iterations, 0.8f0)
               
               codeword = zeros(Int, n)
               received = simulate_bsc(codeword, p)
               llr = llr_from_bsc(received, p)
               
               decode(cpu_decoder, llr)
               decode(gpu_decoder, llr)
               CUDA.synchronize()
               
               cpu_times = []
               for _ in 1:5
                   GC.gc()
                   t_start = time()
                   decode(cpu_decoder, llr)
                   t_end = time()
                   push!(cpu_times, (t_end - t_start) * 1000) 
               end
               
               gpu_times = []
               for _ in 1:5
                   t_start = time()
                   decode(gpu_decoder, llr)
                   CUDA.synchronize() 
                   t_end = time()
                   push!(gpu_times, (t_end - t_start) * 1000) 
               end
               
               sort!(cpu_times)
               sort!(gpu_times)
               avg_cpu_time = mean(cpu_times[1:end-1])
               avg_gpu_time = mean(gpu_times[1:end-1])
               
               speedup = avg_cpu_time / avg_gpu_time
               
               results[n] = (
                   rate = rate,
                   cpu_time = avg_cpu_time,
                   gpu_time = avg_gpu_time,
                   speedup = speedup
               )
               
               @printf("| %10d | %10.3f | %15.3f | %15.3f | %10.2fx |\n", 
                      n, rate, avg_cpu_time, avg_gpu_time, speedup)
           end
           
           return results
       end
       
       """
           benchmark_cpu_vs_gpu_iterations(n, wr, wc, p_values)
       
       Benchmark how decoder performance changes with number of iterations.
       Tests both CPU and GPU decoders with different fixed iteration counts.
       
       Parameters:
       - n: Code size
       - wr: Row weight
       - wc: Column weight
       - p_values: Array of error probabilities to test
       
       Returns:
       - results: Dictionary with benchmark results
       """
       function benchmark_cpu_vs_gpu_iterations(n, wr, wc, p, iteration_values)
           println("\n")
           println("="^60)
           println("ITERATION COUNT BENCHMARK: CPU vs GPU")
           println("="^60)
           println("Code size: $n, Row weight: $wr, Column weight: $wc")
           println("Error rate: $p")
           println("-"^60)
           
           @printf("| %10s | %15s | %15s | %10s |\n", 
                  "Iterations", "CPU Time (ms)", "GPU Time (ms)", "Speedup")
           println("|" * "-"^12 * "|" * "-"^17 * "|" * "-"^17 * "|" * "-"^12 * "|")
           
           H = parity_check_matrix(n, wr, wc)
           H_sparse = sparse(Float32.(H))
           
           codeword = zeros(Int, n)
           received = simulate_bsc(codeword, p)
           llr = llr_from_bsc(received, p)
           
           results = Dict()
           
           for iterations in iteration_values
               cpu_decoder = LDPCDecoderCPU(H_sparse, iterations, 0.8f0)
               gpu_decoder = LDPCDecoder(H_sparse, iterations, 0.8f0)
               
               decode(cpu_decoder, llr)
               decode(gpu_decoder, llr)
               CUDA.synchronize()
               
               cpu_times = []
               for _ in 1:3
                   GC.gc()
                   t_start = time()
                   decode(cpu_decoder, llr)
                   t_end = time()
                   push!(cpu_times, (t_end - t_start) * 1000)  
               end
               
               # doing best of 3
               gpu_times = []
               for _ in 1:3
                   t_start = time()
                   decode(gpu_decoder, llr)
                   CUDA.synchronize()  
                   t_end = time()
                   push!(gpu_times, (t_end - t_start) * 1000)  
               end
               
               min_cpu_time = minimum(cpu_times)
               min_gpu_time = minimum(gpu_times)
               
               speedup = min_cpu_time / min_gpu_time
               
               results[iterations] = (
                   cpu_time = min_cpu_time,
                   gpu_time = min_gpu_time,
                   speedup = speedup
               )
               
               @printf("| %10d | %15.3f | %15.3f | %10.2fx |\n", 
                      iterations, min_cpu_time, min_gpu_time, speedup)
           end
           
           return results
       end
       
       function run_decoding_benchmarks()
           println("RUNNING DECODING-ONLY BENCHMARKS")
           println("These benchmarks focus purely on the decoding algorithm performance")
           
           size_results = benchmark_decoding_only([128, 256, 512, 1024, 2048, 4096], 8, 4, 0.03)
           
           fixed_iter_results = benchmark_fixed_iterations([512, 1024, 2048, 4096], 8, 4, 0.05, 20)
           
           iteration_results = benchmark_cpu_vs_gpu_iterations(2048, 8, 4, 0.05, [1, 2, 5, 10, 20, 50])
           
           return (
               size_results = size_results,
               fixed_iter_results = fixed_iter_results,
               iteration_results = iteration_results
           )
       end
       

        using Random
        using SparseArrays
        using DelimitedFiles
        using Statistics
        using Printf
        using StatsBase: sample
        using CUDA
        

        
        """
            benchmark_accuracy_and_performance(n_values, wr, wc, p_values, num_trials=100)
        
        Benchmark both the decoding accuracy and performance of CPU vs GPU LDPC decoders.
        
        Parameters:
        - n_values: Array of code sizes to test
        - wr: Row weight
        - wc: Column weight  
        - p_values: Array of error probabilities to test
        - num_trials: Number of trials for each configuration
        
        Returns:
        - results: Dictionary with benchmark results
        """
        function benchmark_accuracy_and_performance(n_values, wr, wc, p_values, num_trials=100)
            println("="^70)
            println("ACCURACY AND PERFORMANCE BENCHMARK: CPU vs GPU")
            println("="^70)
            println("Row weight (wr): $wr, Column weight (wc): $wc")
            println("Number of trials per configuration: $num_trials")
            
            results = Dict()
            
            for n in n_values
                if n % wr != 0 || (n * wc) % wr != 0
                    println("\nSkipping invalid configuration: n=$n, wr=$wr, wc=$wc")
                    continue
                end
                
                println("\n" * "-"^70)
                println("CODE SIZE: $n bits, Rate: $(1 - (n*wc)/(wr*n))")
                println("-"^70)
                
                @printf("| %8s | %13s | %13s | %10s | %10s | %9s | %9s | %10s |\n", 
                       "Error %", "CPU Time (ms)", "GPU Time (ms)", "Speedup", "Agreement", "CPU FER", "GPU FER", "CPU=GPU FER")
                println("|" * "-"^10 * "|" * "-"^15 * "|" * "-"^15 * "|" * "-"^12 * "|" * "-"^12 * "|" * "-"^11 * "|" * "-"^11 * "|" * "-"^12 * "|")
                
                H = parity_check_matrix(n, wr, wc)
                H_sparse = sparse(Float32.(H))
                
                cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
                gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
                
                for p in p_values
                    cpu_times = Float64[]
                    gpu_times = Float64[]
                    cpu_successes = 0
                    gpu_successes = 0
                    agreement_count = 0
                    both_correct = 0
                    
                    seeds = rand(UInt32, num_trials)
                    
                    for trial in 1:num_trials
                        Random.seed!(seeds[trial])
                        
                        codeword = zeros(Int, n)
                        
                        received = simulate_bsc(codeword, p)
                        llr = llr_from_bsc(received, p)
                        
                        t_start = time()
                        cpu_decoded, cpu_success, cpu_iterations = decode(cpu_decoder, llr)
                        t_end = time()
                        push!(cpu_times, (t_end - t_start) * 1000)  # Convert to ms
                        cpu_successes += cpu_success ? 1 : 0
                        
                        t_start = time()
                        gpu_decoded, gpu_success, gpu_iterations = decode(gpu_decoder, llr)
                        CUDA.synchronize()  
                        t_end = time()
                        push!(gpu_times, (t_end - t_start) * 1000)  # Convert to ms
                        gpu_successes += gpu_success ? 1 : 0
                        
                        if all(cpu_decoded .== gpu_decoded)
                            agreement_count += 1
                            if all(cpu_decoded .== codeword)
                                both_correct += 1
                            end
                        end
                    end
                    
                    avg_cpu_time = mean(cpu_times)
                    avg_gpu_time = mean(gpu_times)
                    
                    speedup = avg_cpu_time / avg_gpu_time
                    
                    cpu_fer = 1.0 - (cpu_successes / num_trials)
                    gpu_fer = 1.0 - (gpu_successes / num_trials)
                    agreement_rate = agreement_count / num_trials
                    both_correct_rate = both_correct / num_trials
                    
                    if !haskey(results, n)
                        results[n] = Dict()
                    end
                    results[n][p] = (
                        cpu_time = avg_cpu_time,
                        gpu_time = avg_gpu_time,
                        speedup = speedup,
                        agreement_rate = agreement_rate,
                        cpu_fer = cpu_fer,
                        gpu_fer = gpu_fer,
                        both_correct_rate = both_correct_rate
                    )
                    
                    @printf("| %7.3f%% | %13.3f | %13.3f | %10.2fx | %10.2f%% | %9.2f%% | %9.2f%% | %10.2f%% |\n", 
                           p*100, avg_cpu_time, avg_gpu_time, speedup, agreement_rate*100, 
                           cpu_fer*100, gpu_fer*100, both_correct_rate*100)
                end
            end
            
            return results
        end
        
        """
            benchmark_bit_error_rates(n, wr, wc, p_values, num_trials=500)
        
        Benchmark the bit error rates (BER) of CPU and GPU decoders.
        This gives a more detailed view of error correction performance than frame error rate.
        
        Parameters:
        - n: Code size
        - wr: Row weight
        - wc: Column weight
        - p_values: Array of error probabilities to test
        - num_trials: Number of trials for each configuration
        
        Returns:
        - results: Dictionary with BER benchmark results
        """
        function benchmark_bit_error_rates(n, wr, wc, p_values, num_trials=500)
            println("\n" * "="^70)
            println("BIT ERROR RATE BENCHMARK: CPU vs GPU")
            println("="^70)
            println("Code size: $n bits, Row weight: $wr, Column weight: $wc")
            println("Number of trials per configuration: $num_trials")
            println("-"^70)
            
            @printf("| %8s | %10s | %13s | %13s | %12s | %12s |\n", 
                   "Error %", "Raw BER %", "CPU BER %", "GPU BER %", "CPU Gain (dB)", "GPU Gain (dB)")
            println("|" * "-"^10 * "|" * "-"^12 * "|" * "-"^15 * "|" * "-"^15 * "|" * "-"^14 * "|" * "-"^14 * "|")
            
            H = parity_check_matrix(n, wr, wc)
            H_sparse = sparse(Float32.(H))
            
            cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
            gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
            
            results = Dict()
            
            for p in p_values
                total_bits = n * num_trials
                raw_errors = 0
                cpu_errors = 0
                gpu_errors = 0
                
                seeds = rand(UInt32, num_trials)
                
                for trial in 1:num_trials
                    Random.seed!(seeds[trial])
                    
                    codeword = zeros(Int, n)
                    
                    received = simulate_bsc(codeword, p)
                    raw_errors += sum(received) 
                    llr = llr_from_bsc(received, p)
                    
                    cpu_decoded, _, _ = decode(cpu_decoder, llr)
                    cpu_errors += sum(cpu_decoded .!= codeword)
                    
                    gpu_decoded, _, _ = decode(gpu_decoder, llr)
                    gpu_errors += sum(gpu_decoded .!= codeword)
                end
                
                raw_ber = raw_errors / total_bits
                cpu_ber = cpu_errors / total_bits
                gpu_ber = gpu_errors / total_bits
                
               
                cpu_gain_db = cpu_ber > 0 ? 10 * log10(raw_ber / cpu_ber) : Inf
                gpu_gain_db = gpu_ber > 0 ? 10 * log10(raw_ber / gpu_ber) : Inf
                
                results[p] = (
                    raw_ber = raw_ber,
                    cpu_ber = cpu_ber,
                    gpu_ber = gpu_ber,
                    cpu_gain_db = cpu_gain_db,
                    gpu_gain_db = gpu_gain_db
                )
                
                @printf("| %7.3f%% | %10.6f%% | %13.6f%% | %13.6f%% | %12.2f | %12.2f |\n", 
                       p*100, raw_ber*100, cpu_ber*100, gpu_ber*100, cpu_gain_db, gpu_gain_db)
            end
            
            return results
        end
        
        """
            benchmark_error_correction_threshold(n, wr, wc, p_start, p_end, p_step, num_trials=100)
        
        Determine the error correction threshold of the LDPC code.
        This is the maximum error probability at which the decoder can still reliably correct errors.
        
        Parameters:
        - n: Code size
        - wr: Row weight
        - wc: Column weight
        - p_start: Starting error probability
        - p_end: Ending error probability
        - p_step: Step size for error probability
        - num_trials: Number of trials for each configuration
        
        Returns:
        - cpu_threshold: Estimated error correction threshold for CPU decoder
        - gpu_threshold: Estimated error correction threshold for GPU decoder
        """
        function benchmark_error_correction_threshold(n, wr, wc, p_start, p_end, p_step, num_trials=100)
            println("\n" * "="^70)
            println("ERROR CORRECTION THRESHOLD: CPU vs GPU")
            println("="^70)
            println("Code size: $n bits, Row weight: $wr, Column weight: $wc")
            println("Error probability range: $p_start to $p_end (step: $p_step)")
            println("Number of trials per configuration: $num_trials")
            println("-"^70)
            
            @printf("| %8s | %11s | %11s |\n", "Error %", "CPU Success %", "GPU Success %")
            println("|" * "-"^10 * "|" * "-"^13 * "|" * "-"^13 * "|")
            
            H = parity_check_matrix(n, wr, wc)
            H_sparse = sparse(Float32.(H))
            
            cpu_decoder = LDPCDecoderCPU(H_sparse, 50, 0.8f0)
            gpu_decoder = LDPCDecoder(H_sparse, 50, 0.8f0)
            
            results = Dict()
            cpu_threshold = 0.0
            gpu_threshold = 0.0
            cpu_threshold_found = false
            gpu_threshold_found = false
            
            p_values = p_start:p_step:p_end
            
            for p in p_values
                cpu_successes = 0
                gpu_successes = 0
                
                seeds = rand(UInt32, num_trials)
                
                for trial in 1:num_trials
                    Random.seed!(seeds[trial])
                    
                    codeword = zeros(Int, n)
                    
                    received = simulate_bsc(codeword, p)
                    llr = llr_from_bsc(received, p)
                    
                    _, cpu_success, _ = decode(cpu_decoder, llr)
                    cpu_successes += cpu_success ? 1 : 0
                    
                    _, gpu_success, _ = decode(gpu_decoder, llr)
                    gpu_successes += gpu_success ? 1 : 0
                end
                
                cpu_success_rate = cpu_successes / num_trials
                gpu_success_rate = gpu_successes / num_trials
                
                results[p] = (
                    cpu_success_rate = cpu_success_rate,
                    gpu_success_rate = gpu_success_rate
                )
                
                @printf("| %7.3f%% | %11.2f%% | %11.2f%% |\n", 
                       p*100, cpu_success_rate*100, gpu_success_rate*100)
                
                if !cpu_threshold_found && cpu_success_rate < 0.5
                    cpu_threshold = p - p_step  # noteL:The last value where success rate was >= 50%
                    cpu_threshold_found = true
                end
                
                if !gpu_threshold_found && gpu_success_rate < 0.5
                    gpu_threshold = p - p_step  # noteL: The last value where success rate was >= 50%
                    gpu_threshold_found = true
                end
            end
            
            if !cpu_threshold_found
                cpu_threshold = p_end
            end
            
            if !gpu_threshold_found
                gpu_threshold = p_end
            end
            
            println("\nEstimated error correction thresholds:")
            println("  CPU: $(cpu_threshold*100)%")
            println("  GPU: $(gpu_threshold*100)%")
            
            return cpu_threshold, gpu_threshold, results
        end
        
        function run_accuracy_benchmarks()
            println("RUNNING ACCURACY AND PERFORMANCE BENCHMARKS")
            println("These benchmarks compare both the speed and accuracy of CPU and GPU decoders")
            
            accuracy_results = benchmark_accuracy_and_performance(
                [512, 1024, 2048], 
                8, 4,             
                [0.01, 0.03, 0.05, 0.07, 0.09], 
                100                
            )
            
            ber_results = benchmark_bit_error_rates(
                1024, 8, 4,       
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], 
                200                
            )
            
            cpu_threshold, gpu_threshold, threshold_results = benchmark_error_correction_threshold(
                1024, 8, 4,       
                0.04, 0.12, 0.01,  
                100              
            )
            
            return (
                accuracy_results = accuracy_results,
                ber_results = ber_results,
                threshold_results = (
                    cpu_threshold = cpu_threshold,
                    gpu_threshold = gpu_threshold,
                    data = threshold_results
                )
            )
        end
        
            run_accuracy_benchmarks()
        