"""
Accelerated SBM Sampler with Applicant Clustering
James Yuming Yu, 28 November 2024 
with optimizations by Jonah Heyl, Kieran Weaver and others
"""

module SBM

using JSON, Random, Distributions, PrettyTables, HTTP, SparseArrays

export doit, get_placements, bucket_extract, get_results, estimate_parameters, nice_table, api_to_adjacency, parse_placements

function bucket_estimate(assign, A, A_indices, A_vals, T, count, num_applicants, numtier, numtotal)
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
                @inbounds L += base * log(base / (count[i] * count[j]))
            end
        end
    end
    return -L, T
end

function doit(sample, num_applicants, num_departments, department_allocation, sink_counters, numtier, numtotal, blankcount_tol)
    """
        Compute the maximum likelihood estimate of the type assignment via SBM applied to the data `sample`
    """
    
    # some initial states
    cur_objective = Inf
    T = zeros(Int32, numtotal, numtier)
    count = zeros(Int32, numtotal)
    sample_indices = rowvals(sample)
    sample_vals = nonzeros(sample)

    blankcount = 0
    current_allocation = Vector{Int32}(undef, num_applicants + num_departments + sum(sink_counters))
    new_tier_lookup = [deleteat!(Vector(1:numtier), i) for i in 1:numtier]
    cursor = 1

    for _ in 1:num_applicants
        current_allocation[cursor] = 1
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
        @inbounds count[old_tier] -= 1
        @inbounds count[new_tier] += 1
        # check if the new assignment is better
        test_objective, placement_rates = SBM.bucket_estimate(current_allocation, sample, sample_indices, sample_vals, T, count, num_applicants, numtier, numtotal)
        if test_objective < cur_objective
            print("$test_objective ")
            # keep the improvement and continue
            blankcount = 0
            cur_objective = test_objective
            if counter % 100 == 0 
                println()
                println()
                display(placement_rates)
                println()
            end
            counter += 1
        else
            # revert the change
            @inbounds current_allocation[k] = old_tier
            @inbounds count[new_tier] -= 1
            @inbounds count[old_tier] += 1
            # EARLY STOP: if no improvements are possible at all, stop the sampler
            blankcount += 1
            if blankcount % blankcount_tol == 0
                found = false
                for i in 1:num_to_reallocate, tier in 1:numtier
                    # conduct a single-department edit
                    @inbounds original = current_allocation[i]
                    @inbounds current_allocation[i] = tier
                    @inbounds count[original] -= 1
                    @inbounds count[tier] += 1
                    test_objective, placement_rates = SBM.bucket_estimate(current_allocation, sample, sample_indices, sample_vals, T, count, num_applicants, numtier, numtotal)
                    # revert the edit after computing the objective so the allocation is not tampered with
                    @inbounds current_allocation[i] = original
                    @inbounds count[tier] -= 1
                    @inbounds count[original] += 1
                    if test_objective < cur_objective
                        found = true
                        println("continue ")
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

    department_name_mapping = Dict()
    reverse_mapping = Dict()
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
                push!(department_set, string(placement["from_oid"])) 
                from_name = string(placement["from_institution_name"], " - ", placement["from_department"])
                push!(academic, from_name)
                to_name = string(placement["to_name"], " - ", placement["to_department"])
                #institution_mapping[string(placement["from_institution_id"])] = placement["from_institution_name"]
                department_name_mapping[string(placement["from_oid"])] = from_name
                department_name_mapping[string(placement["to_oid"])] = to_name
                # some institutions may have more than one ID. one ID will be *arbitrarily* picked to represent it:
                #reverse_mapping[placement["from_institution_name"]] = string(placement["from_institution_id"])
                
                applicant_department_mapping[applicant_id] = string(placement["from_oid"])
                
                """ NOTE: the OIDs do not have unique "to_name"s, need to include the department name """
                reverse_mapping[to_name] = string(placement["to_oid"]) # this may be redundant now
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

    return applicant_mapping, department_mapping, applicant_list, department_list, academic, academic_to, academic_builder, rough_sink_builder, department_name_mapping, reverse_mapping, applicant_department_mapping
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
            outcome_to_name = string(outcome["to_name"], " - ", outcome["to_department"])
            if !(outcome_to_name in academic)
                out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[string(outcome["from_oid"])]] += 1
            else
                out[department_mapping[string(outcome["to_oid"])], department_mapping[string(outcome["from_oid"])]] += 1
            end
        end
        for (outcome_to_name, outcome) in sink_builder
            outcome_counter += 1
            out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[string(outcome["from_oid"])]] += 1
        end
    else
        for index in indices_to_include
            if index <= length(academic_builder)
                applicant_id, outcome = academic_builder[index]
                outcome_counter += 1
                outcome_to_name = string(outcome["to_name"], " - ", outcome["to_department"])
                if !(outcome_to_name in academic)
                    out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[string(outcome["from_oid"])]] += 1
                else
                    out[department_mapping[string(outcome["to_oid"])], department_mapping[string(outcome["from_oid"])]] += 1
                end
            else
                outcome_to_name, outcome = sink_builder[index - length(academic_builder)]
                outcome_counter += 1
                out[length(department_mapping) + sink_mapping[outcome_to_name], department_mapping[string(outcome["from_oid"])]] += 1
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

function get_adjacency(applicant_mapping, department_mapping, sink_mapping, academic, academic_builder, sink_builder; DEBUG = true, bootstrap_samples = 0)
    """
        Construct the adjacency matrix of placements for clustering by departments and applicants.
    """

    # replace academic_list with applicant_list
    # use university IDs instead of university names
    # map applicant IDs to indices
    out = zeros(Int32, length(department_mapping) + length(sink_mapping), length(applicant_mapping))

    # if bootstrap sampling, only select a few placements
    # if not, this list will be empty (bootstrap_samples = 0)
    indices_to_include = rand(1:length(academic_builder)+length(sink_builder), bootstrap_samples)
    outcome_counter = 0

    if bootstrap_samples == 0
        for (applicant_id, outcome) in academic_builder
            outcome_counter += 1
            outcome_to_name = string(outcome["to_name"], " - ", outcome["to_department"])
            if !(outcome_to_name in academic)
                out[length(department_mapping) + sink_mapping[outcome_to_name], applicant_mapping[applicant_id]] += 1
            else
                out[department_mapping[string(outcome["to_oid"])], applicant_mapping[applicant_id]] += 1
            end
        end
        for (outcome_to_name, outcome) in sink_builder
            outcome_counter += 1
            out[length(department_mapping) + sink_mapping[outcome_to_name], applicant_mapping[outcome["aid"]]] += 1
        end
    else
        for index in indices_to_include
            if index <= length(academic_builder)
                applicant_id, outcome = academic_builder[index]
                outcome_counter += 1
                outcome_to_name = string(outcome["to_name"], " - ", outcome["to_department"])
                if !(outcome_to_name in academic)
                    out[length(department_mapping) + sink_mapping[outcome_to_name], applicant_mapping[applicant_id]] += 1
                else
                    out[department_mapping[string(outcome["to_oid"])], applicant_mapping[applicant_id]] += 1
                end
            else
                outcome_to_name, outcome = sink_builder[index - length(academic_builder)]
                outcome_counter += 1
                out[length(department_mapping) + sink_mapping[outcome_to_name], applicant_mapping[outcome["aid"]]] += 1
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

#=
NUMBER_OF_TYPES: The number of academic types.
BLANKCOUNT_TOL: After the algorithm sees X iterations with no improvements,
it will attempt to check if absolutely no improvements are possible at all.
This parameter is X.
=#

function estimate_parameters(NUMBER_OF_TYPES, BLANKCOUNT_TOL; SEED=0, DEBUG=false)
    """
        Compute parameter estimates via SBM from the `doit` function
    """

    Random.seed!(SEED)         # for reproducibility: ensures random results are the same on script restart
    YEAR_INTERVAL = 2003:2021  # change this to select the years of data to include in the estimation
    NUMBER_OF_SINKS = 4        # this should not change unless you change the sink structure
    numtotal = NUMBER_OF_TYPES + NUMBER_OF_SINKS

    out, academic_list, acd_sink_list, gov_sink_list, pri_sink_list, tch_sink_list, sinks, institutions, institution_mapping = get_placements(YEAR_INTERVAL, DEBUG)
    placement_rates = zeros(Int32, numtotal, NUMBER_OF_TYPES)
    counts = zeros(Int32, numtotal, NUMBER_OF_TYPES)

    est_obj = nothing
    est_alloc = nothing
    if DEBUG
        @time est_obj, est_alloc = doit(out, placement_rates, counts, length(academic_list), length(acd_sink_list), length(gov_sink_list), length(pri_sink_list), length(tch_sink_list), length(institutions), NUMBER_OF_TYPES, numtotal, BLANKCOUNT_TOL)
    else
        est_obj, est_alloc = doit(out, placement_rates, counts, length(academic_list), length(acd_sink_list), length(gov_sink_list), length(pri_sink_list), length(tch_sink_list), length(institutions), NUMBER_OF_TYPES, numtotal, BLANKCOUNT_TOL)
    end

    est_mat, est_count, full_likelihood = bucket_extract(est_alloc, out, NUMBER_OF_TYPES, numtotal)

    sorted_allocation, o = get_results(placement_rates, counts, est_mat, est_count, est_alloc, institutions, NUMBER_OF_TYPES, numtotal)

    if DEBUG
        if NUMBER_OF_TYPES > 1 && all([!(i in est_alloc) for i in 2:NUMBER_OF_TYPES])
            println()
            println("ERROR IN SAMPLER (no movement detected)")
            println()
        else
            for sorted_type in 1:NUMBER_OF_TYPES
                counter = 0
                inst_hold = []
                println("TYPE $sorted_type:")
                for (i, sbm_type) in enumerate(est_alloc)
                    if sorted_type == o[sbm_type]
                        push!(inst_hold, institutions[i])
                        counter += 1
                    end
                end
                for inst in sort(inst_hold)
                    println("  ", inst)
                end
                println("Total Institutions: $counter")
                println()
            end
        end

        try
            mkdir(".estimates")
        catch
        end

        open(".estimates/estimated_raw_placement_counts.json", "w") do f
            write(f, JSON.string(est_mat))
        end

        println("Estimated Placement Counts (unsorted):")
        display(est_mat)
        println()

        for i in 1:NUMBER_OF_TYPES, j in 1:NUMBER_OF_TYPES
            if i > j # not a diagonal and only check once
                if placement_rates[i, j] <= placement_rates[j, i]
                    println("FAULT: hiring ", i, " with graduating ", j, ": downward rate: ", placement_rates[i, j], ", upward rate: ", placement_rates[j, i])
                end
            end
        end
        open(".estimates/estimated_sorted_placement_counts.json", "w") do f
            write(f, JSON.string(placement_rates))
        end
        println()
        println("Estimated Placement Counts (sorted types):")
        display(placement_rates)
        println()
        open(".estimates/estimated_placement_rates.json", "w") do f
            write(f, JSON.string(placement_rates ./ counts))
        end
        println("Estimated Placement Rates (sorted types):")
        display(placement_rates ./ counts)
        println()

        println("Likelihood: $full_likelihood")
        println()
        println("Check Complete")
    end

    return (; 
        placements = placement_rates,
        counters = counts, 
        means = placement_rates ./ counts, 
        likelihood = full_likelihood, 
        allocation = sorted_allocation, 
        institutions,
        num_institutions = length(institutions),
        institution_mapping
    )
end

## print a nice version of the adjacency matrix with tiers and return the latex
## function by Mike Peters from https://github.com/michaelpetersubc/mapinator/blob/355ad808bddcb392388561d25a63796c81ff04c0/estimation/functions.jl
## TODO: port the API functionality from the same file
function nice_table(t_table, num, numsinks, sinks; has_unassigned = false)
    sink_names = [s for s in sinks]
    push!(sink_names, "Column Totals")
    column_sums = sum(t_table, dims=1)
    row_sums = sum(t_table, dims=2)
    row_sums_augmented = vcat(row_sums, sum(row_sums))
    part = vcat(t_table,column_sums)
    
    headers = [""]
    names = []
    
    for i=1:num-1
        push!(headers, "Tier $i")
        push!(names, "Tier $i")
    end
    
    if has_unassigned == false # regular case
        push!(headers, "Tier $num")
        push!(names, "Tier $num")
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

function idcheck(outcome)
    """
        Institution ID conversion helper function for sinks
    """

    if outcome["recruiter_type"] == 5
        return outcome["to_institution_id"] + 5000000
    elseif outcome["recruiter_type"] in [6, 7]
        return outcome["to_institution_id"] + 6000000
    elseif outcome["recruiter_type"] == 8
        return outcome["to_institution_id"] + 8000000
    end
    return outcome["to_institution_id"]
end

function fetch_api(endpoint)
    """
        Extract the relevant placement outcomes from the mapinator API
    """

    api_data = HTTP.get("https://support.econjobmarket.org/api/$(endpoint)")
    return JSON.parse(String(api_data.body))
end

function api_to_adjacency(to_from, tier_data, YEAR_INTERVAL)
    """
        Use the to_from outcomes and tier_data allocation to construct the Poisson means matrix
    """

    # institution IDs
    academic = Dict{}()
    academic_to = Dict{}()

    # placements
    academic_builder = []
    rough_sink_builder = []

    for placement in to_from
        if in(placement["year"], YEAR_INTERVAL)
            academic[parse(Int, placement["from_institution_id"])] = placement["from_institution_name"]
            if placement["position_name"] == "Assistant Professor"
                academic_to[placement["to_institution_id"]] = placement["to_name"]
                push!(academic_builder, placement)
            else
                push!(rough_sink_builder, placement)
            end
        end
    end

    tch_sink = Dict{}() # sink of teaching universities that do not graduate PhDs
    for key in keys(academic_to)
        if !(key in keys(academic))
            tch_sink[key] = string(academic_to[key], " (teaching university)")
        end
    end

    acd_sink = Dict{}()
    gov_sink = Dict{}()
    pri_sink = Dict{}()
    sink_builder = []

    for outcome in rough_sink_builder
        if outcome["recruiter_type"] == 5
            # government institution
            gov_sink[outcome["to_institution_id"]+5000000] = string(outcome["to_name"], " (public sector)")
            push!(sink_builder, outcome)
        elseif outcome["recruiter_type"] in [6, 7]
            # private sector: for and not for profit
            pri_sink[outcome["to_institution_id"]+6000000] = string(outcome["to_name"], " (private sector)")
            push!(sink_builder, outcome)
        elseif outcome["recruiter_type"] == 8
            # international organizations and think tanks
            acd_sink[outcome["to_institution_id"]+8000000] = string(outcome["to_name"], " (international sink)")
            push!(sink_builder, outcome)
        end
    end

    # sort to ensure consistent ordering
    academic_list = sort(collect(keys(academic)))
    acd_sink_list = sort(collect(keys(acd_sink)))
    gov_sink_list = sort(collect(keys(gov_sink)))
    pri_sink_list = sort(collect(keys(pri_sink)))
    tch_sink_list = sort(collect(keys(tch_sink)))
    sinks = vcat(acd_sink_list, gov_sink_list, pri_sink_list, tch_sink_list)
    institutions = vcat(academic_list, sinks)

    institution_names = merge(academic, acd_sink, gov_sink, pri_sink, tch_sink)

    out = zeros(Int32, length(institutions), length(academic_list))
    for outcome in academic_builder
        out[findfirst(isequal(outcome["to_institution_id"]), institutions), findfirst(isequal(parse(Int, outcome["from_institution_id"])), institutions)] += 1
    end
    for outcome in sink_builder
        out[findfirst(isequal(idcheck(outcome)), institutions), findfirst(isequal(parse(Int, outcome["from_institution_id"])), institutions)] += 1
    end

    api_allocation = Dict{}()
    for entry in tier_data
        api_allocation[entry["institution_id"]] = entry["type"]
    end

    NUMBER_OF_TYPES = maximum(values(api_allocation)) + 1 # temporarily accomodate departments with no type allocation (fix TODO)
    NUMBER_OF_SINKS = 4
    numtotal = NUMBER_OF_TYPES + NUMBER_OF_SINKS

    est_alloc = Vector{Int32}(undef, length(institutions))
    cursor = 1
    for institution_id in academic_list
        if institution_id in keys(api_allocation)
            est_alloc[cursor] = api_allocation[institution_id]
        else
            est_alloc[cursor] = NUMBER_OF_TYPES
        end
        cursor += 1
    end
    for _ in acd_sink_list # international organizations
        est_alloc[cursor] = NUMBER_OF_TYPES + 1
        cursor += 1
    end
    for _ in gov_sink_list # govt institutions
        est_alloc[cursor] = NUMBER_OF_TYPES + 2
        cursor += 1
    end
    for _ in pri_sink_list # private sector
        est_alloc[cursor] = NUMBER_OF_TYPES + 3
        cursor += 1
    end
    for _ in tch_sink_list # teaching institutions
        est_alloc[cursor] = NUMBER_OF_TYPES + 4
        cursor += 1
    end

    est_mat, est_count, full_likelihood = bucket_extract(est_alloc, out, NUMBER_OF_TYPES, numtotal)

    #=
    for sorted_type in 1:numtotal
        counter = 0
        println("TYPE $sorted_type:")
        inst_hold = []
        for (i, sbm_type) in enumerate(est_alloc)
            if sorted_type == sbm_type
                push!(inst_hold, institution_names[institutions[i]])
                counter += 1
            end
        end
        sort!(inst_hold)
        for iname in inst_hold
            println("  ", iname)
        end
        println("Total Institutions: $counter")
        println()
    end
    =#

    return est_mat ./ est_count # just the means
end

end