#!/usr/bin/julia

function calc_ind_block(global_size::Int, local_size::Int, local_features::Int, num_features::Int, num_clusters::Int)
    work_groups = div(global_size, local_size);
    group_ids = repeat(collect(0:(work_groups - 1)), inner = [local_size], outer = [1]); 
    global_ids = collect(0:(global_size - 1));
    local_ids = repmat(collect(0:(local_size - 1)), work_groups, 1);

    centroids_size = num_features * num_clusters;
    block_size = div(num_features, local_features);
    block = div(local_ids, block_size);
    num_blocks = div(local_size, block_size);
    num_local_points = local_size;
    num_block_points = div(num_local_points, num_blocks);
    tile = div(global_ids, block_size);
    block_offset = num_clusters * num_features * block;
    tile_offset = num_clusters * num_features * (group_ids * num_blocks + block);
    l_feature = mod(local_ids, block_size);

    println("block_size           : $block_size")
    println("num_blocks           : $num_blocks")
    println("num_local_points     : $num_local_points")
    println("num_block_points     : $num_block_points")
    println("block                : $(block')")
    println("tile                 : $(tile')")
    println("block_offset         : $(block_offset')")
    println("tile_offset          : $(tile_offset')")
    println("l_feature            : $(l_feature')")
end
