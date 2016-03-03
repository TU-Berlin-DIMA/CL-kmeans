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

function calc_ind_old(global_size::Int, local_size::Int, num_features::Int, num_clusters::Int)
    opt_global_size = div_rnd_up(num_features * num_clusters, local_size) * local_size;
    work_groups = div(global_size, local_size);
    group_ids = repeat(collect(0:(work_groups - 1)), inner = [local_size], outer = [1]); 
    global_ids = collect(0:(global_size - 1));
    local_ids = repmat(collect(0:(local_size - 1)), work_groups, 1);

    local_cols = max(1, div(local_size, num_clusters));
    local_rows_points = div(local_size, local_cols);
    local_rows_clusters = min(num_clusters, local_rows_points);
    global_blks = div(global_size, (local_rows_points * local_cols));
    g_blk = div(group_ids * local_size, (local_rows_points * local_cols));
    tile_cols = div_rnd_up(num_features, local_cols);
    tile_rows = div_rnd_up(num_clusters, local_rows_clusters);
    l_row_points = local_ids % local_rows_points;
    l_row_clusters = local_ids % local_rows_clusters;
    l_col = div(local_ids, local_rows_points);
    t_row = div(global_ids, local_rows_clusters) % tile_rows;
    t_col = div(global_ids, (local_cols * local_rows_points * tile_rows)) % tile_cols;
    b_row = l_row_clusters + local_rows_clusters * t_row;
    b_col = l_col + local_cols * t_col;
    println("optimal global size: $opt_global_size")
    println("work groups        : $work_groups")
    println("local rows clusters: $local_rows_clusters")
    println("local rows points  : $local_rows_points")
    println("local cols         : $local_cols")
    println("global blks        : $global_blks")
    println("tile rows          : $tile_rows")
    println("tile cols          : $tile_cols")
    println("group id           : $(group_ids')")
    println("g_blk              : $(g_blk')")
    println("b_row              : $(b_row')")
    println("b_col              : $(b_col')")
    println("l_row_clusters     : $(l_row_clusters')")
    println("l_row_points       : $(l_row_points')")
    println("l_col              : $(l_col')")
    println("t_row              : $(t_row')")
    println("t_col              : $(t_col')")
end

function calc_ind_block(global_size::Int, local_size::Int, num_features::Int, num_clusters::Int)
    opt_global_size = div_rnd_up(num_features * num_clusters, local_size) * local_size;
    work_groups = div(global_size, local_size);
    group_ids = repeat(collect(0:(work_groups - 1)), inner = [local_size], outer = [1]); 
    global_ids = collect(0:(global_size - 1));
    local_ids = repmat(collect(0:(local_size - 1)), work_groups, 1);

    centroids_size = num_features * num_clusters;
    num_local_blocks = div(local_size, centroids_size);
    num_global_blocks = div(global_size, centroids_size);
    l_block = div(local_ids, centroids_size);
    g_block = div(global_ids, centroids_size);
    l_pos = local_ids - l_block * centroids_size;
    l_cluster = l_pos % num_clusters;
    l_feature = div(l_pos, num_clusters);
    cache_num_points = div(local_size, num_features);
    cache_point = local_ids % cache_num_points;
    cache_feature = div(local_ids, cache_num_points);
    g_num_points = div(global_size, num_features);
    g_point = cache_point + group_ids * cache_num_points;
    l_num_points = div(cache_num_points, num_local_blocks);
    l_point_begin = l_block * l_num_points;

    println("num_local_blocks     : $num_local_blocks")
    println("num_global_blocks    : $num_global_blocks")
    println("l_block              : $(l_block')")
    println("g_block              : $(g_block')")
    println("l_pos                : $(l_pos')")
    println("l_cluster            : $(l_cluster')")
    println("l_feature            : $(l_feature')")
    println("cache_num_points     : $(cache_num_points')")
    println("cache_point          : $(cache_point')")
    println("cache_feature        : $(cache_feature')")
    println("g_num_points         : $(g_num_points)")
    println("g_point              : $(g_point')")
    println("l_point_begin        : $(l_point_begin')")
end
