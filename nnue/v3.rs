use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::{LocalSettings, TestDataset},
    },
    value::{ValueTrainerBuilder, loader}, // loader::DirectSequentialDataLoader,
};

// --- env overrides (used by py training wrapper) ---
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i16(key: &str, default: i16) -> i16 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}



fn main() {
    // hyperparams to fiddle with
    let l1_size = env_usize("L1", 1024);
    let l2_size = env_usize("L2", 16);
    let l3_size = env_usize("L3", 32);

    let qa_16: i16 = env_i16("QA", 127);
    let qa_32: i32 = env_i32("QA", 127);
    let qb_16: i16 = env_i16("QB", 64);
    let qb_32: i32 = env_i32("QB", 64);
    let qc_16: i16 = env_i16("QC", 64);
    let qc_32: i32 = env_i32("QC", 64);

    let initial_lr          = env_f32("lr_start", 0.001);
    let final_lr            = env_f32("lr_final", initial_lr * 0.3f32.powi(5));
    let superbatches        = env_usize("superbatches", 800);
    let superbatch_start    = env_usize("superbatch_start", 0);
    let wdl_start           = env_f32("wdl_start", 0.25);
    let wdl_end             = env_f32("wdl_end", 0.25);

    let net_id              = env_string("net_id", "1024x16x32_25wdl");
    let output_dir          = env_string("output_dir", "checkpoints");
    let train_path          = env_string("train_data", "data/T60T70wIsRightFarseer.binpack");
    let val_path            = env_string("val_data", "data/test80-2023-12-dec-2tb7p.min-v2.v6.binpack");

    let _save_rate          = env_usize("save_rate", 10);
    let _batch_size         = env_usize("batch_size", 16_384);
    let _batches            = env_usize("batches", 6104);
    let _threads            = env_usize("threads", 4);

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
    const NUM_OUTPUT_BUCKETS: usize = 8;
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ];


    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(qa_16),
            SavedFormat::id("l0b").round().quantise::<i16>(qa_16),
            SavedFormat::id("l1w").round().quantise::<i8>(qb_16),
            SavedFormat::id("l1b").round().quantise::<i32>(qa_32 * qb_32),
            SavedFormat::id("l2w").round().quantise::<i8>(qc_16),
            SavedFormat::id("l2b").round().quantise::<i32>(qa_32 * qb_32 * qc_32),
            SavedFormat::id("l3w").round().quantise::<i8>(qc_16),
            SavedFormat::id("l3b").round().quantise::<i32>(qa_32 * qb_32 * qc_32 * qc_32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(l1_size, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, l1_size);
            //l0.init_with_effective_input_size(l3);
            l0.weights = l0.weights + l0f.repeat(NUM_INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", l1_size, NUM_OUTPUT_BUCKETS * l2_size);
            let l2 = builder.new_affine("l2", l2_size, NUM_OUTPUT_BUCKETS * l3_size);
            let l3 = builder.new_affine("l3", l3_size, NUM_OUTPUT_BUCKETS);

            // Faster version of
            // let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            // let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            
            // NOTE: crelu() not screlu()
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, l1_size / 2) * ft(stm_inputs, l1_size / 2, l1_size);
            let ntm_hidden = ft(ntm_inputs, 0, l1_size / 2) * ft(ntm_inputs, l1_size / 2, l1_size);

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: _batch_size,
            batches_per_superbatch: _batches,
            start_superbatch: superbatch_start,
            end_superbatch: superbatch_start + superbatches - 1,
        },
        wdl_scheduler: wdl::LinearWDL { start: wdl_start, end: wdl_end },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: _save_rate,
    };

    // ========================================================
    // Training + validation datasets
    // ========================================================

    use loader::sfbinpack::SfBinpackLoader;

    
    fn filter(entry: &loader::sfbinpack::TrainingDataEntry) -> bool {
        use loader::sfbinpack::{MoveType,PieceType,};
        entry.ply >= 16
            && !entry.pos.is_checked(entry.pos.side_to_move())
            && entry.score.unsigned_abs() <= 10000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type()== PieceType::None
    }

    let train_loader =
        SfBinpackLoader::new(
            &train_path,
            1024,
            _threads,
            filter,
        );

    let val_loader =
        SfBinpackLoader::new(
            &val_path,
            256,
            _threads/2,
            filter,
        );

    // ========================================================
    // Trainer settings
    // ========================================================

    let settings = LocalSettings {
        threads: _threads,
        test_set: Some(
            TestDataset::at(&val_path)
            .freq(_batches) // once per superbatch
            .batches(256), // ~ 4 million validation positions (64 * _batch_size)
        ),
        output_directory: &output_dir,
        batch_queue_size: 64,
    };

    // ========================================================
    // Train with validation
    // ========================================================

    // cirriculum learning
    //  if this is not superbatch 1, then we are loading from a trained net
    //  so we load the last checkpoint (start-1) and continue
    if superbatch_start > 1 {
        let checkpoint = format!(
            "{}/{}-{}",
            output_dir,
            net_id,
            superbatch_start - 1
        );

        trainer.load_from_checkpoint(&checkpoint);
    }

    trainer.run_with_validation(
        &schedule,
        &settings,
        &train_loader,
        &val_loader,
    );

}
