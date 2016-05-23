#!/usr/bin/julia

#=
Keep in mind:

assert( b_col < num_features )
assert( b_row < num_clusters )
assert( g_blk < global_blks )
assert( point_offset + l_row < num_points )
=#

function div_rnd_up(dividend::Int, divisor::Int)
    return div((dividend + divisor - 1), divisor)
end

function calc_ind_block(global_size::Int, local_size::Int, num_features::Int, num_clusters::Int, round::Int)
    opt_global_size = div_rnd_up(num_features * num_clusters, local_size) * local_size;
    work_groups = div(global_size, local_size);
    group_ids = repeat(collect(0:(work_groups - 1)), inner = [local_size], outer = [1]); 
    global_ids = collect(0:(global_size - 1));
    local_ids = repmat(collect(0:(local_size - 1)), work_groups, 1);

    group_base_a = group_ids * local_size * 2 * 2 ^ round;
    group_base_b = (group_ids * local_size * 2 + local_size) * 2 ^ round;
    item_a = group_base_a + local_ids;
    item_b = group_base_b + local_ids;

    println("group ids            : $(group_ids')")
    println("group_base_a         : $(group_base_a')")
    println("group_base_b         : $(group_base_b')")
    println("item_a               : $(item_a')")
    println("item_b               : $(item_b')")
end
