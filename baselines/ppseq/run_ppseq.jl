"""
Run PP-Seq (Williams, Degleris, Wang & Linderman, 2020) on one spike file.

    julia --project=<this dir> run_ppseq.jl <spikes.tsv> <num_neurons> <max_time> <config.json> <out.json> [masked]

spikes.tsv: two tab-separated columns, neuron index (1-based) and time (model time units).
config.json: the PP-Seq config dictionary (keys as in the PPSeq.jl demos; strings are
converted to symbols). With the optional 5th argument "masked", a speckled hold-out
(config "percent_masked", "mask_lengths") is applied and the held-out log-likelihood of
the post-anneal samples is reported, following the authors' cross-validation procedure.

Output JSON (full run): per saved post-anneal sample, the latent events (time, type,
amplitude) and the globals of the final sample (neuron offsets, widths, response
proportions per type, type proportions, background rates); plus the log-joint history.
"""

import PPSeq
const seq = PPSeq
import JSON
import DelimitedFiles: readdlm
import Random

function load_config(path)
    raw = JSON.parsefile(path)
    cfg = Dict{Symbol,Any}()
    for (k, v) in raw
        cfg[Symbol(k)] = v isa Integer ? Int64(v) : (v isa AbstractFloat ? Float64(v) : v)
    end
    # integers that the sampler indexes with
    for k in (:num_sequence_types, :num_warp_values, :num_anneals, :samples_per_anneal, :save_every_during_anneal,
              :samples_after_anneal, :save_every_after_anneal, :split_merge_moves_during_anneal,
              :split_merge_moves_after_anneal, :num_spike_resamples_per_anneal, :num_spike_resamples,
              :samples_per_resample)
        haskey(cfg, k) && (cfg[k] = Int64(cfg[k]))
    end
    if haskey(cfg, :max_sequence_length) && cfg[:max_sequence_length] isa AbstractString
        cfg[:max_sequence_length] = Inf
    end
    return cfg
end

function main(args)
    spikes_path, num_neurons, max_time, config_path, out_path = args[1], parse(Int, args[2]), parse(Float64, args[3]), args[4], args[5]
    masked = length(args) >= 6 && args[6] == "masked"
    cfg = load_config(config_path)

    spikes = seq.Spike[]
    for row in eachrow(readdlm(spikes_path, '\t', Float64, '\n'))
        push!(spikes, seq.Spike(Int(row[1]), row[2]))
    end
    init = fill(-1, length(spikes))
    model = seq.construct_model(cfg, max_time, num_neurons)
    t0 = time()

    if masked
        # The mask is drawn from its own seed so that every candidate hyperparameter setting is
        # validated on the same held-out spikes; the sampler then gets the run seed.
        Random.seed!(get(cfg, :mask_seed, 12345))
        masks = seq.create_random_mask(num_neurons, max_time + 1e-6, cfg[:mask_lengths] + 1e-9, cfg[:percent_masked])
        masks = seq.merge_contiguous_masks(masks, num_neurons)
        Random.seed!(get(cfg, :seed, 0))
        res = seq.easy_sample_masked!(model, spikes, masks, init, cfg)
        test = res[:test_log_p_hist]
        train = res[:train_log_p_hist]
        n_held = length(seq.split_spikes_by_mask(spikes, masks)[1])
        half = (length(test) ÷ 2 + 1):length(test)
        events_hist = res[:latent_event_hist]
        final_assign = res[:assignment_hist][:, end]
        out = Dict(
            "mode" => "masked",
            # the package's statistic: (log-likelihood of the held-out spikes minus that of a homogeneous
            # Poisson process on the same masked neuron-time volume) / masked neuron-time volume,
            # averaged over post-anneal samples
            "heldout_log_like" => sum(test[half]) / max(length(half), 1),
            "heldout_log_like_hist" => test,
            "train_log_like_hist" => train,
            "n_heldout_spikes" => n_held,
            "n_latent_events_hist" => [length(ev) for ev in events_hist],
            "n_background_final" => count(==(-1), final_assign),
            "n_unmasked_spikes" => length(final_assign),
            "runtime_s" => time() - t0,
        )
        open(out_path, "w") do io
            JSON.print(io, out)
        end
        return
    end

    Random.seed!(get(cfg, :seed, 0))
    res = seq.easy_sample!(model, spikes, init, cfg)
    g = res[:globals_hist][end]
    saved_events = [[Dict("t" => e.timestamp, "type" => e.seq_type, "amp" => e.amplitude, "warp" => e.seq_warp) for e in ev]
                    for ev in res[:latent_event_hist]]
    final_assign = res[:assignment_hist][:, end]
    final_events = res[:latent_event_hist][end]
    # spikes per sequence type in the final sample
    type_of_event = Dict(e.assignment_id => e.seq_type for e in final_events)
    spikes_per_type = zeros(Int, cfg[:num_sequence_types])
    for a in final_assign
        a > 0 && haskey(type_of_event, a) && (spikes_per_type[type_of_event[a]] += 1)
    end
    out = Dict(
        "mode" => "full",
        "num_neurons" => num_neurons,
        "max_time" => max_time,
        "neuron_offsets" => g.neuron_response_offsets,            # N x R
        "neuron_widths" => g.neuron_response_widths,              # N x R (variances)
        "neuron_response_proportions" => exp.(g.neuron_response_log_proportions),
        "seq_type_proportions" => exp.(g.seq_type_log_proportions),
        "bkgd_amplitude" => g.bkgd_amplitude,
        "bkgd_proportions" => exp.(g.bkgd_log_proportions),
        "saved_events" => saved_events,
        "final_events" => saved_events[end],
        "spikes_per_type_final" => spikes_per_type,
        "n_background_final" => count(==( -1), final_assign),
        "log_p_hist" => res[:log_p_hist],
        "runtime_s" => time() - t0,
    )
    open(out_path, "w") do io
        JSON.print(io, out)
    end
end

length(ARGS) >= 5 && main(ARGS)
