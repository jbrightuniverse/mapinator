"""
Accelerated SBM Sampler with Double Clustering
James Yuming Yu, 26 February 2025
with optimizations by Jonah Heyl, Kieran Weaver and others
"""

module SBM

using JSON, Random, Distributions, PrettyTables, HTTP, SparseArrays

export doit, bucket_extract, get_results, nice_table

function bucket_estimate(assign, A, A_indices, A_vals, T, count_hiring, count_academic, num_applicants, numtier, numtotal)
    """
        Estimate the simplified log-likelihood of the data `A` given type assignment guess `assign`
    """
    @inbounds T .= 0
    #@inbounds count .= 0
    L = 0.0

    
    for j in 1:size(A)[2]
        @simd for i_raw in nzrange(A, j)
            i = A_indices[i_raw]
            @inbounds val = numtotal * (assign[j] - 1) + assign[num_applicants + i]
            @inbounds T[val] += A_vals[i_raw]
            #@inbounds count[val] += 1
        end
    end

    for i in 1:numtotal
        @simd for j in 1:numtier
            @inbounds base = T[i, j]
            if base != 0.0
                @inbounds L += base * log(base / (count_hiring[i] * count_academic[j]))
            end
        end
    end
    return -L, T
end

function doit(sample, num_applicants, num_departments, department_allocation, sink_counters, numtier, numtotal, blankcount_tol; should_print = true)
    """
        Compute the maximum likelihood estimate of the type assignment via SBM applied to the data `sample`
    """
    
    # some initial states
    T = zeros(Int32, numtotal, numtier)
    count = zeros(Int32, numtotal)
    count_academic = zeros(Int32, numtier)
    sample_indices = rowvals(sample)
    sample_vals = nonzeros(sample)

    blankcount = 0
    current_allocation = Vector{Int32}(undef, num_applicants + num_departments + sum(sink_counters))
    new_tier_lookup = [deleteat!(Vector(1:numtier), i) for i in 1:numtier]
    cursor = 1

    for _ in 1:num_applicants
        current_allocation[cursor] = 1
        count_academic[1] += 1
        cursor += 1
    end
    # TODO: try hardcoding the department tiers

    if isnothing(department_allocation)
        for _ in 1:num_departments
            current_allocation[cursor] = 1
            count[1] += 1
            cursor += 1
        end
    else
        for tier in department_allocation
            current_allocation[cursor] = tier
            count[tier] += 1
            cursor += 1
        end
    end

    for (i, num_sink) in enumerate(sink_counters)
        for _ in 1:num_sink
            current_allocation[cursor] = numtier + i # the sinks must stay in fixed types
            count[numtier + i] += 1
            cursor += 1
        end
    end

    count_to_pass = nothing
    if num_applicants > 0
        count_to_pass = count_academic
    else
        count_to_pass = count
    end
    cur_objective, _ = SBM.bucket_estimate(current_allocation, sample, sample_indices, sample_vals, T, count, count_to_pass, num_applicants, numtier, numtotal)
    
    if numtier == 1 # if only one tier is required, we are already done
        return cur_objective, current_allocation
    end

    if isnothing(department_allocation)
        num_to_reallocate = num_applicants+num_departments
    else
        num_to_reallocate = num_applicants
    end

    counter = 0
    while true # BEGIN MONTE CARLO REALLOCATION: attempt to reallocate applicants and academic institutions to a random spot
        k = rand(1:num_to_reallocate)
        @inbounds old_tier = current_allocation[k]
        @inbounds new_tier = rand(new_tier_lookup[old_tier])
        @inbounds current_allocation[k] = new_tier
        if num_applicants > 0 && k <= num_applicants
            @inbounds count_academic[old_tier] -= 1
            @inbounds count_academic[new_tier] += 1
        else
            @inbounds count[old_tier] -= 1
            @inbounds count[new_tier] += 1
        end
        count_to_pass = nothing
        if num_applicants > 0
            count_to_pass = count_academic
        else
            count_to_pass = count
        end
        # check if the new assignment is better
        test_objective, placement_rates = SBM.bucket_estimate(current_allocation, sample, sample_indices, sample_vals, T, count, count_to_pass, num_applicants, numtier, numtotal)
        if test_objective < cur_objective
            if should_print print("$test_objective ") end
            # keep the improvement and continue
            blankcount = 0
            cur_objective = test_objective
            if should_print && (counter % 100 == 0) 
                println()
                println()
                display(placement_rates)
                println()
            end
            counter += 1
        else
            # revert the change
            @inbounds current_allocation[k] = old_tier
            if num_applicants > 0 && k <= num_applicants
                @inbounds count_academic[new_tier] -= 1
                @inbounds count_academic[old_tier] += 1
            else
                @inbounds count[new_tier] -= 1
                @inbounds count[old_tier] += 1
            end
            # EARLY STOP: if no improvements are possible at all, stop the sampler
            blankcount += 1
            if blankcount % blankcount_tol == 0
                found = false
                for i in 1:num_to_reallocate, tier in 1:numtier
                    # conduct a single-department edit
                    @inbounds original = current_allocation[i]
                    @inbounds current_allocation[i] = tier
                    if num_applicants > 0 && i <= num_applicants
                        @inbounds count_academic[original] -= 1
                        @inbounds count_academic[tier] += 1
                    else
                        @inbounds count[original] -= 1
                        @inbounds count[tier] += 1
                    end
                    count_to_pass_early = nothing
                    if num_applicants > 0
                        count_to_pass_early = count_academic
                    else
                        count_to_pass_early = count
                    end
                    test_objective, placement_rates = SBM.bucket_estimate(current_allocation, sample, sample_indices, sample_vals, T, count, count_to_pass_early, num_applicants, numtier, numtotal)
                    # revert the edit after computing the objective so the allocation is not tampered with
                    @inbounds current_allocation[i] = original
                    if num_applicants > 0 && i <= num_applicants
                        @inbounds count_academic[tier] -= 1
                        @inbounds count_academic[original] += 1
                    else
                        @inbounds count[tier] -= 1
                        @inbounds count[original] += 1
                    end
                    if test_objective < cur_objective
                        found = true
                        if should_print println("continue ") end
                        break
                    end
                end
                if !found
                    return cur_objective, current_allocation
                end
            end
        end
    end
end

function keycheck(outcome)
    """
        Department name conversion helper function for sinks
    """

    if outcome["recruiter_type"] == 5
        return string(outcome["to_name"], " (public sector)")
    elseif outcome["recruiter_type"] in [6, 7]
        return string(outcome["to_name"], " (private sector)")
    elseif outcome["recruiter_type"] == 8
        return string(outcome["to_name"], " (international sink)")
    end
    return outcome["to_name"]
end

function fetch_api(endpoint)
    """Retrieve data from a URL endpoint."""
    api_data = HTTP.get(endpoint) # e.g. "https://support.econjobmarket.org/api/placement_data"
    res = JSON.parse(String(api_data.body))
    if "error" in keys(res)
        error(string(res["error"], ": ", res["error_description"]))
    end
    return res
end

function api_to_placements(endpoint)
    """Sort the placement data from an API endpoint by placement year."""
    to_from_by_year = DefaultDict(Dict)
    to_from = fetch_api(endpoint)
    for placement in to_from
        to_from_by_year[string(placement["year"])][string(placement["aid"])] = placement
    end
    return to_from_by_year
end

function json_to_placements(endpoint)
    """Get placement data from a filepath."""
    return JSON.parsefile(endpoint) # e.g. "to_from_by_year.json"
end

function fetch_data(endpoint)
    """
        Get placement data from an arbitrary endpoint.
    """

    if startswith(endpoint, "http") # a URL
        return api_to_placements(endpoint)
    else
        return json_to_placements(endpoint) # a JSON file
    end
end

function get_builders(to_from_by_year, YEAR_INTERVAL)
    """
        Extract the placement outcomes occurring in YEAR_INTERVAL
    """

    academic = Set()
    academic_to = Set()
    academic_builder = []
    rough_sink_builder = []

    applicant_department_mapping = Dict()

    applicant_set = Set()
    department_set = Set()

    for year in keys(to_from_by_year)
        if parse(Int32, year) in YEAR_INTERVAL
            # ASSUMPTION: every applicant ID has only one placement
            for (applicant_id, placement) in to_from_by_year[year]
                # IDs of the applicants and departments
                push!(applicant_set, applicant_id)
                # NOTE: the sets may not have the same order when converted to lists
                from_name = placement["from_institution_name"]
                push!(department_set, from_name)
                push!(academic, from_name)
                to_name = placement["to_name"]

                applicant_department_mapping[applicant_id] = placement["from_institution_name"]
                
                if placement["position_name"] == "Assistant Professor"
                    push!(academic_to, to_name)
                    push!(academic_builder, (applicant_id, placement))
                else
                    push!(rough_sink_builder, placement)
                end
            end
        end
    end

    applicant_list = collect(applicant_set)
    applicant_mapping = Dict()
    for (i, applicant) in enumerate(applicant_list)
        applicant_mapping[applicant] = i
    end

    department_list = collect(department_set)
    department_mapping = Dict()
    for (i, department) in enumerate(department_list)
        department_mapping[department] = i
    end

    return applicant_mapping, department_mapping, applicant_list, department_list, academic, academic_to, academic_builder, rough_sink_builder, applicant_department_mapping
end

function build_sinks(sinks_to_include, teaching_list; DEBUG = true)
    """
        Construct the sink departments based on the sink placements.
    """
    
    sink_builder = []
    sinks = []
    sink_labels = []

    if DEBUG 
        println("Including the following sinks:") 
    end
    for (sink_name, sink_placements) in sinks_to_include
        if DEBUG 
            println(" ", sink_name) 
        end
        # generate list of department names
        dept_names = Set()
        for (outcome_to_name, outcome) in sink_placements
            push!(dept_names, outcome_to_name)
            push!(sink_builder, (outcome_to_name, outcome))
        end
        push!(sinks, sort(collect(dept_names)))
        push!(sink_labels, sink_name)
    end

    push!(sinks, teaching_list) # teaching_list must always be included
    push!(sink_labels, "Teaching Universities")
    if DEBUG 
        println(" Teaching Universities")
        println("Total $(length(sink_labels)) sinks")
    end

    sink_list = vcat(sinks...)
    sink_mapping = Dict()
    for (i, sink) in enumerate(sink_list)
        sink_mapping[sink] = i
    end

    return sink_mapping, sink_list, sink_builder, sinks, sink_labels
end

function get_adjacency_by_department(department_mapping, sink_mapping, academic, academic_builder, sink_builder; DEBUG = true, bootstrap_samples = 0)
    """
        Construct the adjacency matrix of placements for clustering by departments only.
    """

    out = zeros(Int32, length(department_mapping) + length(sink_mapping), length(department_mapping))

    # if bootstrap sampling, only select a few placements
    # if not, this list will be empty (bootstrap_samples = 0)
    indices_to_include = rand(1:length(academic_builder)+length(sink_builder), bootstrap_samples)
    outcome_counter = 0

    if bootstrap_samples == 0
        for (applicant_id, outcome) in academic_builder
            outcome_counter += 1
            outcome_to_name = outcome["to_name"]
            if !(outcome_to_name in academic)
                out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
            else
                out[department_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
            end
        end
        for (outcome_to_name, outcome) in sink_builder
            outcome_counter += 1
            out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
        end
    else
        for index in indices_to_include
            if index <= length(academic_builder)
                applicant_id, outcome = academic_builder[index]
                outcome_counter += 1
                outcome_to_name = outcome["to_name"]
                if !(outcome_to_name in academic)
                    out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
                else
                    out[department_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
                end
            else
                outcome_to_name, outcome = sink_builder[index - length(academic_builder)]
                outcome_counter += 1
                out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[outcome["from_institution_name"]]] += 1
            end
        end
    end

    if DEBUG
        println("Total ", length(academic_builder)+length(sink_builder), " Placements (found ", outcome_counter, " by sequence counting, ", sum(out), " by matrix sum)")
        if bootstrap_samples != 0
            println(" bootstrapping ", bootstrap_samples, " samples")
        end
    end

    return out
end


function bucket_extract(assign, A, num_applicants, numtier, numtotal)
    """
        Extract the Poisson means from data `A` given type assignment `assign`
    """

    b = zeros(Int32, size(A))
    T = zeros(Int32, numtotal, numtier)
    count = zeros(Int32, numtotal, numtier)
    for i in 1:size(A)[1], j in 1:size(A)[2]
        @inbounds val = numtotal * (assign[j] - 1) + assign[num_applicants + i]
        @inbounds b[i, j] = val
        @inbounds T[val] += A[i, j]
        @inbounds count[val] += 1
    end
    
    L = 0.0
    @simd for i in eachindex(A)
        @inbounds L += logpdf(Poisson(T[b[i]]/count[b[i]]), A[i])
    end
    return T, count, L
end

function get_results(est_mat, est_count, est_alloc, num_applicants, NUMBER_OF_TYPES, numtotal, sort_tiers)
    """
        Compiles sorted SBM results
    """

    placement_rates = zeros(Int32, numtotal, NUMBER_OF_TYPES)
    counts = zeros(Int32, numtotal, NUMBER_OF_TYPES)

    # mapping o[i]: takes an unsorted SBM-marked type i and outputs the corresponding true, sorted type
    if sort_tiers == 1 # graduating and hiring tiers are the same
        o1 = zeros(Int32, numtotal)
        o1[vcat(sortperm(vec(sum(est_mat, dims = 1)), rev=true), NUMBER_OF_TYPES+1:numtotal)] = 1:numtotal
        o2 = o1
    elseif sort_tiers == 2 # graduating and hiring tiers are different
        o1 = zeros(Int32, numtotal)
        o1[vcat(sortperm(vec(sum(est_mat, dims = 1)), rev=true), NUMBER_OF_TYPES+1:numtotal)] = 1:numtotal
        o2 = zeros(Int32, numtotal)
        o2[vcat(sortperm(vec(sum(est_mat[1:NUMBER_OF_TYPES, :], dims = 2)), rev=true), NUMBER_OF_TYPES+1:numtotal)] = 1:numtotal
    else # no sorting
        o1 = 1:numtotal
        o2 = o1
    end

    # shuffle the cells for the tier to tier placements
    for i in 1:numtotal
        @simd for j in 1:NUMBER_OF_TYPES
            placement_rates[o2[i], o1[j]] = est_mat[i, j]
            counts[o2[i], o1[j]] = est_count[i, j]
        end
    end

    # shuffle the allocation
    sorted_allocation = Vector{Int32}(undef, length(est_alloc))
    for (i, entry) in enumerate(est_alloc)
        if i <= num_applicants
            sorted_allocation[i] = o1[entry]
        else
            sorted_allocation[i] = o2[entry]
        end
    end

    return placement_rates, counts, sorted_allocation, o1, o2
end

function get_allocation(est_alloc, out, num_applicants, NUMBER_OF_TYPES, numtotal; sort_tiers = 1)
    """
        Compile the results of an SBM allocation.
    """

    est_mat, est_count, full_likelihood = SBM.bucket_extract(est_alloc, out, num_applicants, NUMBER_OF_TYPES, numtotal)
    placement_rates, counts, sorted_allocation, o1, o2 = SBM.get_results(est_mat, est_count, est_alloc, num_applicants, NUMBER_OF_TYPES, numtotal, sort_tiers)
    return placement_rates, counts, sorted_allocation, full_likelihood
end

## print a nice version of the adjacency matrix with tiers and return the latex
## function by Mike Peters from https://github.com/michaelpetersubc/mapinator/blob/355ad808bddcb392388561d25a63796c81ff04c0/estimation/functions.jl
## TODO: port the API functionality from the same file
function nice_table(t_table, numtier, numhiring, numsinks, sinks; has_unassigned = false)
    sink_names = [s for s in sinks]
    push!(sink_names, "Column Totals")
    column_sums = sum(t_table, dims=1)
    row_sums = sum(t_table, dims=2)
    row_sums_augmented = vcat(row_sums, sum(row_sums))
    part = vcat(t_table,column_sums)
    
    headers = [""]
    names = []
    
    for i in 1:numtier-1
        push!(headers, "Tier $i")
    end

    for i in 1:numhiring-1
        push!(names, "Tier $i")
    end
    
    if has_unassigned == false # regular case
        push!(headers, "Tier $numtier")
        push!(names, "Tier $numhiring")
    else # need to use an "Unassigned" tier
        push!(headers, "Missing")
        push!(names, "Missing")
    end 
    
    #println(headers)

    push!(headers, "Row Totals")
    names = cat(names, sink_names, dims=1)
    adjacency = hcat(names, part, row_sums_augmented)
    #headers = ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Row totals"]
    #names = ["Tier 1","Tier 2","Tier 3","Tier 4","Other Academic","Government","Private Sector","Teaching Universities","Column Totals"]
    #pretty_table(adjacency, header = headers, row_names=names)
    pretty_table(adjacency, header = headers)
    return pretty_table(adjacency, header = headers, backend=Val(:latex))
end

end