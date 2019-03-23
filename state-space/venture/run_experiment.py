import venture
import venture.shortcuts as vs
import numpy as np
import collections
import json

def convert_from_stack_dict(stack_dict):
    import venture.lite.value as vv
    venture_value = vv.VentureValue.fromStackDict(stack_dict)
    return convert_from_venture_value(venture_value)


def convert_from_venture_value(venture_value):
    """Convert a stack dict to python object."""
    import venture.lite.value as vv
    from venture.lite.types import Dict
    if isinstance(venture_value, vv.VentureDict):
        shallow = Dict().asPythonNoneable(venture_value)
        deep = collections.OrderedDict()
        for key, value in shallow.iteritems():
            deep[convert_from_venture_value(key)] = \
                convert_from_venture_value(value)
        return deep
    elif isinstance(venture_value, vv.VentureNumber):
        return venture_value.getNumber()
    elif isinstance(venture_value, vv.VentureInteger):
        return venture_value.getInteger()
    elif isinstance(venture_value, vv.VentureString):
        return venture_value.getString()
    elif isinstance(venture_value, vv.VentureBool):
        return venture_value.getBool()
    elif isinstance(venture_value, vv.VentureAtom):
        return venture_value.getAtom()
    elif isinstance(venture_value, vv.VentureArray):
        return [
            convert_from_venture_value(val)
            for val in venture_value.getArray()
        ]
    elif isinstance(venture_value, vv.VentureArrayUnboxed):
        return [
            convert_from_venture_value(val)
            for val in venture_value.getArray()
        ]
    elif isinstance(venture_value, vv.VentureMatrix):
        return venture_value.matrix
    elif isinstance(venture_value, vv.VenturePair):
        return [
            convert_from_venture_value(val)
            for val in venture_value.getArray()
        ]
    else:
        raise ValueError(
            'Venture value cannot be converted', str(venture_value))

ripl = vs.make_lite_ripl(seed=1)

ripl.load_plugin("extensions.py")

def execute_venture_program(prog):
    results = ripl.execute_program(prog, type=True)
    return convert_from_stack_dict(results[-1]["value"])

execute_venture_program("""
define times = list(0.0, 0.0526316, 0.105263, 0.157895, 0.210526, 0.263158, 0.315789,
                    0.368421, 0.421053, 0.473684, 0.526316, 0.578947, 0.631579, 0.684211,
                    0.736842, 0.789474, 0.842105, 0.894737, 0.947368, 1.0);

define start_x = 0.1;
define start_y = 0.1;
define stop_x = 0.5;
define stop_y = 0.5;
define speed = 0.5;
define noise = 0.02;
define dist_slack = 0.2;

define path = list(
    pair(0.1, 0.1),
    pair(0.0773627, 0.146073),
    pair(0.167036, 0.655448),
    pair(0.168662, 0.649074),
    pair(0.156116, 0.752046),
    pair(0.104823, 0.838075),
    pair(0.196407, 0.873581),
    pair(0.390309, 0.988468),
    pair(0.408272, 0.91336),
    pair(0.5, 0.5)
);

define distances_from_start = list(0.0, 0.0513339, 0.568542, 0.57512, 0.678854,
                                   0.779013, 0.877239, 1.10262, 1.17985, 1.60326);

define do_particle_filter = (num_particles, x_obs, y_obs, times, speed, noise, dist_slack) -> {
    
    timer = start_timer();
    trace = new_trace();
    
    // load the functions into the trace
    _ = run_in_trace(trace, {
    
        assume walk_path_recurse = (distances_from_start, path_point_index, dist) -> {
            if (dist < distances_from_start[path_point_index]) {
                integer(path_point_index - integer(1))
            } else {
                walk_path_recurse(distances_from_start, integer(path_point_index + integer(1)), dist)
            }
        };

        assume walk_path = (path, distances_from_start, dist) -> {
            if (dist <= 0.0) {
                path[integer(0)]
            } else {
                path_length = (distances_from_start[
                    integer(integer(size(distances_from_start)) - integer(1))]);
                if (dist >= path_length) {
                    path[integer(integer(size(path)) - integer(1))]
                } else {
                    path_point_index = walk_path_recurse(distances_from_start, integer(1), dist);
                    dist_from_path_point = (dist - (distances_from_start[path_point_index]));
                    dist_between_points = (distances_from_start[integer(path_point_index + integer(1))]
                                            - distances_from_start[path_point_index]);
                    fraction_next = dist_from_path_point / dist_between_points;
                    x = ((fraction_next * first(path[integer(path_point_index + integer(1))]))
                                            + ((1.0 - fraction_next) * first(path[path_point_index])));
                    y = ((fraction_next * rest(path[integer(path_point_index + integer(1))]))
                                            + ((1.0 - fraction_next) * rest(path[path_point_index])));
                    pair(x, y)
                }
            }
        };
        });
    
    // spawn number of particles
    _ = run_in_trace(trace, resample(num_particles));
                
    // sample --- 0 ---
    _ = run_in_trace(trace, {
        assume dist0 = normal(${speed} * ${times}[0], ${dist_slack});
        assume loc0 = walk_path(${path}, ${distances_from_start}, dist0);
        observe normal(first(loc0), ${noise}) = x_obs[0];
        observe normal(rest(loc0), ${noise}) = y_obs[0];
    });
    
    // get log weights and resample
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 1 ---
    _ = run_in_trace(trace, {
        assume dist1 = normal(dist0 + (${speed} * (${times}[1] - ${times}[0])), ${dist_slack});
        assume loc1 = walk_path(${path}, ${distances_from_start}, dist1);
        observe normal(first(loc1), ${noise}) = x_obs[1];
        observe normal(rest(loc1), ${noise}) = y_obs[1];
    });
    
    // get log weights and resample
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 2 ---
    _ = run_in_trace(trace, {
        assume dist2 = normal(dist1 + (${speed} * (${times}[2] - ${times}[1])), ${dist_slack});
        assume loc2 = walk_path(${path}, ${distances_from_start}, dist2);
        observe normal(first(loc2), ${noise}) = x_obs[2];
        observe normal(rest(loc2), ${noise}) = y_obs[2];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 3 ---
    _ = run_in_trace(trace, {
        assume dist3 = normal(dist2 + (${speed} * (${times}[3] - ${times}[2])), ${dist_slack});
        assume loc3 = walk_path(${path}, ${distances_from_start}, dist3);
        observe normal(first(loc3), ${noise}) = x_obs[3];
        observe normal(rest(loc3), ${noise}) = y_obs[3];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 4 ---
    _ = run_in_trace(trace, {
        assume dist4 = normal(dist3 + (${speed} * (${times}[4] - ${times}[3])), ${dist_slack});
        assume loc4 = walk_path(${path}, ${distances_from_start}, dist4);
        observe normal(first(loc4), ${noise}) = x_obs[4];
        observe normal(rest(loc4), ${noise}) = y_obs[4];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 5 ---
    _ = run_in_trace(trace, {
        assume dist5 = normal(dist4 + (${speed} * (${times}[5] - ${times}[4])), ${dist_slack});
        assume loc5 = walk_path(${path}, ${distances_from_start}, dist5);
        observe normal(first(loc5), ${noise}) = x_obs[5];
        observe normal(rest(loc5), ${noise}) = y_obs[5];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 6 ---
    _ = run_in_trace(trace, {
        assume dist6 = normal(dist5 + (${speed} * (${times}[6] - ${times}[5])), ${dist_slack});
        assume loc6 = walk_path(${path}, ${distances_from_start}, dist6);
        observe normal(first(loc6), ${noise}) = x_obs[6];
        observe normal(rest(loc6), ${noise}) = y_obs[6];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 7 ---
    _ = run_in_trace(trace, {
        assume dist7 = normal(dist6 + (${speed} * (${times}[7] - ${times}[6])), ${dist_slack});
        assume loc7 = walk_path(${path}, ${distances_from_start}, dist7);
        observe normal(first(loc7), ${noise}) = x_obs[7];
        observe normal(rest(loc7), ${noise}) = y_obs[7];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 8 ---
    _ = run_in_trace(trace, {
        assume dist8 = normal(dist7 + (${speed} * (${times}[8] - ${times}[7])), ${dist_slack});
        assume loc8 = walk_path(${path}, ${distances_from_start}, dist8);
        observe normal(first(loc8), ${noise}) = x_obs[8];
        observe normal(rest(loc8), ${noise}) = y_obs[8];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 9 ---
    _ = run_in_trace(trace, {
        assume dist9 = normal(dist8 + (${speed} * (${times}[9] - ${times}[8])), ${dist_slack});
        assume loc9 = walk_path(${path}, ${distances_from_start}, dist9);
        observe normal(first(loc9), ${noise}) = x_obs[9];
        observe normal(rest(loc9), ${noise}) = y_obs[9];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 10 ---
    _ = run_in_trace(trace, {
        assume dist10 = normal(dist9 + (${speed} * (${times}[10] - ${times}[9])), ${dist_slack});
        assume loc10 = walk_path(${path}, ${distances_from_start}, dist10);
        observe normal(first(loc10), ${noise}) = x_obs[10];
        observe normal(rest(loc10), ${noise}) = y_obs[10];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 11 ---
    _ = run_in_trace(trace, {
        assume dist11 = normal(dist10 + (${speed} * (${times}[11] - ${times}[10])), ${dist_slack});
        assume loc11 = walk_path(${path}, ${distances_from_start}, dist11);
        observe normal(first(loc11), ${noise}) = x_obs[11];
        observe normal(rest(loc11), ${noise}) = y_obs[11];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 12 ---
    _ = run_in_trace(trace, {
        assume dist12 = normal(dist11 + (${speed} * (${times}[12] - ${times}[11])), ${dist_slack});
        assume loc12 = walk_path(${path}, ${distances_from_start}, dist12);
        observe normal(first(loc12), ${noise}) = x_obs[12];
        observe normal(rest(loc12), ${noise}) = y_obs[12];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 13 ---
    _ = run_in_trace(trace, {
        assume dist13 = normal(dist12 + (${speed} * (${times}[13] - ${times}[12])), ${dist_slack});
        assume loc13 = walk_path(${path}, ${distances_from_start}, dist13);
        observe normal(first(loc13), ${noise}) = x_obs[13];
        observe normal(rest(loc13), ${noise}) = y_obs[13];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 14 ---
    _ = run_in_trace(trace, {
        assume dist14 = normal(dist13 + (${speed} * (${times}[14] - ${times}[13])), ${dist_slack});
        assume loc14 = walk_path(${path}, ${distances_from_start}, dist14);
        observe normal(first(loc14), ${noise}) = x_obs[14];
        observe normal(rest(loc14), ${noise}) = y_obs[14];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 15 ---
    _ = run_in_trace(trace, {
        assume dist15 = normal(dist14 + (${speed} * (${times}[15] - ${times}[14])), ${dist_slack});
        assume loc15 = walk_path(${path}, ${distances_from_start}, dist15);
        observe normal(first(loc15), ${noise}) = x_obs[15];
        observe normal(rest(loc15), ${noise}) = y_obs[15];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 16 ---
    _ = run_in_trace(trace, {
        assume dist16 = normal(dist15 + (${speed} * (${times}[16] - ${times}[15])), ${dist_slack});
        assume loc16 = walk_path(${path}, ${distances_from_start}, dist16);
        observe normal(first(loc16), ${noise}) = x_obs[16];
        observe normal(rest(loc16), ${noise}) = y_obs[16];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 17 ---
    _ = run_in_trace(trace, {
        assume dist17 = normal(dist16 + (${speed} * (${times}[17] - ${times}[16])), ${dist_slack});
        assume loc17 = walk_path(${path}, ${distances_from_start}, dist17);
        observe normal(first(loc17), ${noise}) = x_obs[17];
        observe normal(rest(loc17), ${noise}) = y_obs[17];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 18 ---
    _ = run_in_trace(trace, {
        assume dist18 = normal(dist17 + (${speed} * (${times}[18] - ${times}[17])), ${dist_slack});
        assume loc18 = walk_path(${path}, ${distances_from_start}, dist18);
        observe normal(first(loc18), ${noise}) = x_obs[18];
        observe normal(rest(loc18), ${noise}) = y_obs[18];
    });
    _ = run_in_trace(trace, resample(num_particles));

    // sample --- 19 ---
    _ = run_in_trace(trace, {
        assume dist19 = normal(dist18 + (${speed} * (${times}[19] - ${times}[18])), ${dist_slack});
        assume loc19 = walk_path(${path}, ${distances_from_start}, dist19);
        observe normal(first(loc19), ${noise}) = x_obs[19];
        observe normal(rest(loc19), ${noise}) = y_obs[19];
    });
    lw19 = run_in_trace(trace, particle_log_weights());

    elapsed = time_elapsed(timer);

    dict(["log_weights", lw19],
         ["elapsed", elapsed])
};

"nothing"
""")

execute_venture_program("""
define measured_xs = list(0.0896684, 0.148145, 0.123211, 0.11035, 0.148417, 0.185746, 0.175872, 0.178704,
    0.150475, 0.175573, 0.150151, 0.172628, 0.121426, 0.222041, 0.155273, 0.164001,
    0.136586, 0.0687045, 0.146904, 0.163813);

define measured_ys = list(0.217256, 0.416599, 0.376985, 0.383586, 0.500322, 0.608227, 0.632844, 0.653351,
    0.532425, 0.881112, 0.771766, 0.653384, 0.756946, 0.870473, 0.8697, 0.808217,
    0.598147, 0.163257, 0.611928, 0.657514);
""")

execute_venture_program("""
define do_pf = (num_particles, num_reps) -> {
    parallel_mapv((i) -> {            
        do_particle_filter(num_particles, measured_xs, measured_ys, times, speed, noise, dist_slack)
    }, arange(num_reps))
};

"nothing"
""")

def logsumexp(log_x_arr):
    max_log = np.max(log_x_arr)
    return max_log + np.log(np.sum(np.exp(log_x_arr - max_log)))

def lml_estimate(log_weights):
    return logsumexp(log_weights) - np.log(len(log_weights))

all_results = dict()
num_particles_list = [1, 3, 10, 30, 100, 300, 1000, 3000]
for num_particles in num_particles_list:
    print "num_particles={num_particles}".format(num_particles=num_particles)
    results = execute_venture_program("do_pf({num_particles}, {num_reps})".format(num_particles=num_particles, num_reps=96))
    lml_list = []
    elapsed_list = []
    for replicate_results in results:
        log_weights = replicate_results["log_weights"]
        elapsed = replicate_results["elapsed"]
        lml = lml_estimate(log_weights)
        lml_list.append(lml)
        elapsed_list.append(elapsed)
    all_results[num_particles] = { "lmls" : lml_list, "elapsed" : elapsed_list }

    print all_results
    with open('venture_results.json', 'w') as outfile:  
        json.dump(all_results, outfile)
