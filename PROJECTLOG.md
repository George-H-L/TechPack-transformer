# Project Log

Please regularly update this file to record your project progress. You should be updating the project log _at least_ once a fortnight.

## Week 1 [w/c Nov 17, 2025]

The project is running on a local server. At this point the structure is bare - the goal was just to get something running before building outward. The design target is a lofi aesthetic - simple ui like gemini or chatgpt, which suits the kind of tool this is: functional and direct.

## Week 2 [w/c Nov 28, 2025]

Deleted the builder module this week and reconsidered where views, forms and utilities should live. The thinking here was about risk - making decisions about structure too early and then building on top of them tends to cause compounding problems later. Better to pause and get the layout right before writing logic that depends on it. Landed on what should be the final project structure, and committed to it.

## Week 3 [w/c Dec 01, 2025]

Created the URL routing, set up a superuser for the admin panel, and wrote the core model logic for TechPack objects. Also moved the app into ghl5 to tidy up the repo structure - there was a duplicate sitting outside it that wasn't doing anything. Set up the CSS sheet and pushed views logic further. The non-AI path is taking shape.

## Weeks 4-10 [w/c Jan 22, 2026]

A gap in commits here that doesn't reflect a gap in work - a lot of this period was reading, planning the data generation approach, and writing logic that got refactored rather than committed. By the end of this stretch, most of the forms and views were in place. The SVG engine was added along with HTML templates for the create and detail pages. The goal at this point was to get the non-AI path fully working before building the model, so that when inference was integrated there was a known-good baseline to test against.

## Weeks 11-12 [w/c Feb 03, 2026]

Started generating training data through the Claude API (claude-3-5-sonnet). The approach is knowledge distillation - using a large capable model to generate examples that the small custom model learns from. Single-turn data came out well. Two-turn conversations were started but stopped about halfway through because the cost was accumulating faster than expected; rate limits were also causing interruptions, and the sleep timing between calls had to be tuned. Reduced batch sizes and shortened the sleep to 15 seconds to try to stay within the output token limit. Resumed from the most recent checkpoint after one interruption. By the end all data generation was complete, though the flattening function needed fixing before the data was actually usable for training.

## Week 13 [w/c Feb 11, 2026]

Fixed the flattening function and saved the training and validation sets. Built the ML model - a custom encoder-decoder transformer small enough to run on a laptop CPU. The encoder reads the garment description and the decoder generates the structured JSON output. Initial training ran to completion. The SVG generator was partially revamped at the same time, though it wasn't finished.

## Week 14 [w/c Feb 18, 2026]

Small fixes to the detail page. Nothing significant.

## Weeks 15-16 [w/c Mar 03, 2026]

Implemented the modify page. At this point it only has sliders - the AI modification side wasn't ready yet because the model hadn't been trained on modification examples. The sliders let you adjust measurements and material fields after generation, which covers the most common case. The TODO list at this point was significant: more training data, turn-based modification context, text-based modification, and getting bottoms working. Bottoms being absent was a known gap; the training data was entirely tops, so the model had no concept of inseam, rise, leg opening, or any of the measurements that bottoms require.

## Week 22 [w/c Apr 16, 2026]

Generated 1000 synthetic bottoms conversations. The first approach was gpt-oss:20b running locally - a single conversation test took 208 seconds, and a proper trial of 50 conversations took 6364 seconds, around 127 seconds per conversation. At that rate, 1000 conversations would have been roughly 35 hours, and the GPU was struggling under the load. gpt-oss:20b produced the best output quality of the local GPT models tested - 48 of 50 conversations came through correctly structured - but the hardware constraints made it impractical at scale. The summaries for those runs are in synthetic/gptoss/. The decision was made to switch to qwen2.5:7b instead.

Several models were tested before settling on qwen2.5:7b. The test runs are still in the synthetic/ollama directory - test_7b.json, test_7bv2.json and day1_test_v3.json each correspond to a different prompting configuration tried on a small sample before committing to a full run. Qwen2.5:7b produced the most consistent structured output of the options tried; the summaries of each model tested are recorded in those files. The full generation was run in checkpointed batches of 50 conversations at a time, which is why the checkpoint_50 through checkpoint_950 files exist - if the process was interrupted it could resume rather than restart. The target was 1100 but generation stopped at 1000, which was judged sufficient.

The bottoms data covers jeans, trousers, chinos, shorts, joggers, skirts and cargo variants, with the full bottoms measurement schema. The quality is noticeably less consistent than the Claude tops data - some measurements come through as null, and the phrasing is occasionally odd - but it gives the model enough signal to learn the bottoms schema.

## Week 23 [w/c Apr 23, 2026]

Gitlab was down for a couple days so the comitts came a bit later than expected, which is why there is a surplus of work in this week - work was continuing locally but couldn't be pushed. This session was focused on combining the two datasets and extending the model to handle both garment categories. The Claude tops data and the Ollama bottoms data were merged into a single combined training set of roughly 8,900 examples. The Ollama conversations needed flattening first since they were stored as raw multi-turn dicts rather than the flat input/output format the dataset loader expects - each turn becomes its own training example. The whole set was shuffled with a fixed seed and split 90/10 into train and validation.

The tokenizer had to be rebuilt from scratch on the combined corpus. The old tokenizer was trained on tops data only, which meant bottoms-specific terms like inseam, outseam, leg_opening and waistband_height were hitting UNK tokens during encoding, so the model would have had no way to learn them even with the new data. min_freq=1 was used rather than the default of 2 because some bottoms terms appear infrequently enough that raising the threshold would have dropped them. A manual check confirmed all critical terms made it in.

Fine-tuning rather than training from scratch was the right call here. The tops model had already learned to produce valid JSON structure and reasonable measurement ranges for jackets, t-shirts and so on - starting over would have thrown that away. The learning rate was dropped to 1e-5, ten times lower than the original, so the model adjusts toward bottoms without overwriting what it already knows. Embedding layers for the new vocab tokens were extended rather than reset: old rows were copied from the checkpoint, new token rows kept their Xavier initialisation. This means the model starts with working representations for tops tokens and random-but-bounded representations for bottoms tokens, rather than everything being random.

Confidence extraction was added to the inference path. During greedy decoding, the softmax probability of each chosen token is recorded. After generation, these probabilities are mapped back to JSON fields using the token sequence - when a field name token appears, the probabilities of the value tokens that follow it are averaged geometrically to give a confidence score per field. The reason for geometric rather than arithmetic mean is that a single very uncertain token should drag the whole field's confidence down, not be averaged away by the others.

That confidence data feeds a follow-up question system. If the model was uncertain about a field and that field isn't one where the answer is obvious from the garment type - jeans being denim, for instance, or a hoodie having a drawstring - the system asks the user a clarifying question before saving. The important detail here is how the answers are handled: rather than extracting keywords from the user's response and patching them into the dict directly, the answer is combined with the original description and re-run through the model. This means if someone types "terracotta" or "washed indigo" the model processes it in context, rather than failing because those words weren't in a lookup table.

Domain-specific validation was added on top of that. The main checks are: fabric and colour fields being swapped (the model occasionally puts "black" in fabric_type and "cotton" in colour), measurement values outside plausible human ranges being clamped silently, bottoms consistency checks (inseam must be shorter than outseam, and rise plus inseam should roughly equal outseam), fabric weight being a bare number without a unit getting gsm appended, and the details array being normalised to a list of one to four strings. The distinction between what gets fixed silently and what triggers a question was about ambiguity - a swapped fabric and colour field has an obvious fix, an implausible fabric weight description doesn't.

## End of Week 23 

This stretch was almost entirely SVG visual fidelity work, driven by side-by-side comparisons against reference tech pack drawings. I rebuilt the polo collar twice. The first version put the collar leaves above the band like a stand collar, which read as a turtleneck more than a polo. The second pass used the standard shirt collar band as a base and added two trapezoidal leaves spreading down and outward from the bottom of the band, plus a short centre placket with three buttons dropping below where the leaves meet. I also reduced the polo sleeve taper from 16 to 8 to 4 over multiple iterations because the earlier values produced visibly triangular sleeves; the final value keeps the sleeve edges nearly parallel, with the cuff opening width boosted by 1.15×. I then applied the same band-plus-leaves collar to all shirts, with the centre placket suppressed since shirts already have a full-length button placket. I deleted the standalone polo button drawing logic as part of this. Buttons now live inside the collar component.

Bottoms got a 5-pocket pass. Jeans had no visible back pockets despite the construction box claiming a 5-pocket layout, and the existing front "scoop" stitch was a smooth quadratic curve rather than the angular opening real jeans have. My fix was to introduce a shared front pocket helper applied to jeans, chinos, and trousers: a short vertical line drops from the waistband, then a second segment leaves at a 110° interior angle to meet the side seam. I computed the terminating point by intersecting the angled line with the seam parametrically rather than approximating the length. I drew the back patch pockets for jeans as pentagonal shapes (flat top, tapered bottom point) sitting just below the back yoke, matching the standard 5-pocket jeans silhouette. I added a waistband button, always for jeans, and for any other bottom with a button-based closure type. For soft bottoms like joggers and sweatpants, I added side-seam slip pockets traced as a curved stitch line following the seam, and I extended the drawstring trigger so any bottom with a drawstring closure shows the cord, not just garments named like joggers.

I gave shirts a left-chest patch pocket when the construction calls for one. The trigger is the pocket field being non-empty, and it is drawn as a simple square patch with a top stitch line at the wearer's left chest. This was the visible gap on the linen shirt example. The construction box said "chestpocket" but the SVG rendered nothing.

Two smaller follow-ups: I removed the navy wool blazer chip from the example row on the create page since the model has weakest accuracy on blazer-specific fields and I wanted to avoid steering people toward a poor first impression, and I updated the sleeve hem cuff for jumpers to fill with a lighter color, calculated by lightening the fill by 22%, so the hem reads visually instead of disappearing into the body fill.

Finished by implementing Docker to allow for a quick execution.  