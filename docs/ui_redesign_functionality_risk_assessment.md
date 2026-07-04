# UI Redesign Functionality Risk Assessment

This note reviews the proposed Streamlit UI redesign in `~/.claude/plans/generic-mapping-pebble.md` against the current elephant re-identification app. The proposed redesign is directionally strong for the main field-staff task: reviewing AI match proposals with less page switching and fewer visible controls. The main risk is that the plan currently treats several operator and scientist workflows as incidental UI clutter when they are actually part of the data lifecycle.

## Summary Recommendation

Proceed with the three-section redesign, but do not implement it as a strict feature reduction. Keep the simplified default path for field reviewers, and add an explicit Advanced or Scientist mode for project setup, bulk actions, raw evidence, diagnostics, catalog update, and ground-truth analysis.

The redesign should preserve these workflow capabilities even if their visual treatment changes:

- Resume versus restart project state.
- Bulk accept and bulk assign-new-ID actions.
- Recovery of below-threshold matches by accepting an AI candidate.
- Candidate rank cycling and reference-image browsing.
- Raw matching evidence: fused score, global similarity, local inlier count, viewpoint, matched image ID, and ground truth when available.
- Query metadata creation from arbitrary image directories.
- Pipeline stop control and raw log/error access.
- Final catalog update after human review.
- Ground-truth evaluation and error-case inspection.
- Optional single-image and viewpoint inspection.

## Current Workflow Surface

The current app has 10 Streamlit pages wired in `app.py`:

1. Home
2. Create Query Table
3. Preprocess Images
4. Run Reidentification
5. Verify Reidentification
6. Identify Unknown
7. Verify New IDs
8. Update Catalogue
9. Validate Ground Truth
10. Visualize Image

The redesign consolidates these into Home / Project Status, Run Analysis, Review Matches, and Review Unknowns. That simplification is sensible, but some current pages are not optional from a workflow perspective. They create metadata, run long jobs, stop long jobs, save expert labels, update the reference catalog, and validate model performance.

## Functionality at Risk

### 1. Bulk Review Actions

Current capability:

- `Accept All Matched IDs` marks all currently matched query images as accepted.
- `Assign New ID to All` marks all currently not-matched query images as new individuals.

Relevant implementation:

- `st_pages/st_4_verify_reidentification.py`
- `update_human_inputs_all(...)`
- `Accept All Matched IDs`
- `Assign New ID to All`

Risk under redesign:

The proposed card UI has only Accept, Reject, and Skip for one item at a time. That is better for ambiguous review, but it removes the high-throughput path for trusted, high-confidence batches.

Recommendation:

Keep bulk actions in an Advanced drawer or batch toolbar. Gate them behind filters such as high confidence, already reviewed, unreviewed, matched, not matched, or current queue.

### 2. Resume Versus Restart Semantics

Current capability:

The current review page distinguishes between resuming an existing project and starting a new review that clears previous `human_input` values.

Relevant implementation:

- `st_pages/st_4_verify_reidentification.py`
- `initialize_vizualization_project()`
- `Resume Project`
- `Start New Project`

Risk under redesign:

Auto-loading the project removes friction, but it can hide whether the app is using saved human inputs, unsaved session edits, or a reset state. Accidental restart or stale review state would be especially costly because downstream partitioning and catalog update depend on `human_input`.

Recommendation:

Auto-load by default, but show a persistent project-state banner with review source, last saved time, reviewed count, and a guarded `Reset Review Inputs` action. Do not erase `human_input` without an explicit confirmation.

### 3. Below-Threshold Match Recovery

Current capability:

The not-matched review flow lets a reviewer accept an AI candidate anyway and overwrite `matching_status` back to `matched`. This is important when the matching threshold is too strict but the suggested identity is visibly correct.

Relevant implementation:

- `st_pages/st_4_verify_reidentification.py`
- `main_analyze_not_matched_images(...)`
- `Accept AI Matched ID`
- `overwrite_matching_results(...)`

Risk under redesign:

The plan says not-matched images will appear automatically in the same queue after matched images. If they are reduced to the same Accept / Reject / Skip semantics, the special case of accepting a rejected AI candidate may be lost.

Recommendation:

In the not-matched section of the queue, label the action clearly, for example `Accept Candidate Anyway`, and ensure it writes both `human_input = AcceptId` and the appropriate matched status/result fields.

### 4. Candidate Rank Cycling and Reference Image Browsing

Current capability:

The current review page supports:

- Cycling among top-N algorithmic candidates.
- Browsing multiple reference images for the candidate individual.
- Jumping forward and backward through query and reference images.

Relevant implementation:

- `st_pages/st_4_verify_reidentification.py`
- `Next Algorithmic Matched ID`
- `Next Reference Image`
- `Previous Reference Image`
- `Jump Forward 100 Images`
- `Jump Forward 10 Images`

Risk under redesign:

A single query-versus-best-match card is easier to understand, but it can reduce evidence. Field reviewers often need to compare a query against more than one image of the same known individual, especially with pose, occlusion, lighting, and left/right viewpoint variation.

Recommendation:

Keep the default card focused on the top candidate, but retain a candidate strip and a reference gallery. The proposed candidate cycling control is good; make sure it can reveal all recommended ranks and multiple reference photos for each rank.

### 5. Raw Matching Evidence and Scientific Diagnostics

Current capability:

The current UI displays detailed match evidence:

- Matched individual ID.
- Matched image ID.
- Fused score.
- Global similarity.
- Local inlier count.
- Query viewpoint.
- Candidate viewpoint.
- Human input.
- Ground truth when available.
- Keypoint overlay when matching payloads exist.

Relevant implementation:

- `st_pages/st_4_verify_reidentification.py`
- `display_custom_table(...)`
- `display_custom_table_human_input(...)`
- `display_custom_table_ground_truth(...)`
- `try_render_keypoint_overlay(...)`

Risk under redesign:

Replacing raw values with a confidence bar improves readability for field staff but weakens debugging and threshold tuning. The score columns and viewpoint columns are not just presentation details; they support calibration and same-view/cross-view analysis.

Recommendation:

Use the confidence bar in the default view, but keep raw details in an expandable Evidence panel. Preserve the keypoint overlay as a collapsible diagnostic, as the plan already suggests.

### 6. Query Metadata Creation

Current capability:

The Create Query Table page lets a user enter an image directory and generate the required metadata table from the images in that directory.

Relevant implementation:

- `st_pages/st_1_create_query_table.py`
- `process_directory(...)`
- `get_img_paths_from_a_folder(...)`

Risk under redesign:

The plan moves this to Settings/Admin. That is acceptable only if it remains first-class for starting a new survey batch. If hidden too deeply, users may not know how to onboard new images.

Recommendation:

On the dashboard, when no query metadata exists, show a direct call to action: `Create Query Table`. Keep the detailed path input in Settings/Admin.

### 7. Pipeline Control, Stop, and Raw Logs

Current capability:

The pipeline pages can start long-running scripts, stop the active tmux session, detect whether a job is running, and show stdout/stderr.

Relevant implementation:

- `st_pages/st_2_preprocess_images.py`
- `st_pages/st_3_run_reidentification.py`
- `st_pages/st_5_identify_unknown_individuals.py`
- `st_pages/st_7_update_catalogue.py`
- `run_bash_script(...)`
- `terminate_script(...)`
- `check_status(...)`

Risk under redesign:

Progress bars and collapsed logs are more usable than raw terminal output, but the plan should not remove stop controls or raw error access. Long-running image processing and matching jobs need an operator escape hatch.

Recommendation:

The Run Analysis page should include:

- Per-step Run / Re-run.
- Run All.
- Stop current job.
- Running status.
- Parsed progress.
- Collapsed human-readable logs.
- Expandable raw stdout/stderr or downloadable log files.
- Explicit failed-state remediation.

### 8. Final Catalogue Update

Current capability:

The app includes an Update Catalogue page, and the pipeline includes `step_6_update_database.py`, which is responsible for carrying reviewed query results into the reference catalog.

Relevant implementation:

- `app.py`
- `st_pages/st_7_update_catalogue.py`
- `pipeline/step_6_update_database.py`

Risk under redesign:

The proposed Run Analysis sequence lists detect/crop, extract features, run matching, and cluster unknowns. It does not clearly include the final catalog update. Without this step, expert review results may never become part of the future reference catalog.

Recommendation:

Represent catalog update as a separate guarded step after review, not as a hidden admin action. It should show prerequisites, review counts, skipped items, new IDs, accepted known IDs, and a confirmation before writing to the catalog.

### 9. Ground-Truth Evaluation and Error Inspection

Current capability:

The Validate Ground Truth page can compute metrics and inspect specific classes of mistakes:

- Falsely matched in-sample images.
- Falsely matched out-of-sample images.
- Falsely unmatched in-sample images.
- Predicted match versus ground-truth reference image.

Relevant implementation:

- `st_pages/st_8_validate_based_on_ground_truth.py`
- `main_compute_matching_accuracy(...)`
- `find_algorithm_mistakes(...)`
- `main_visualize_known_matched_false(...)`
- `main_visualize_unknown_matched_false(...)`
- `main_visualize_fn_table(...)`

Risk under redesign:

The plan says this is kept as an optional Admin/Scientist view. That is the right placement, but it should not be reduced to metrics only. The visual error review is the useful part for model debugging.

Recommendation:

Keep a Scientist Evaluation page with both metrics and visual error-case review. Include same-view versus cross-view metrics because elephant matching depends heavily on viewpoint.

### 10. Single-Image and Viewpoint Inspection

Current capability:

The Visualize Image page lets users inspect a specific image and optionally look up its viewpoint tag from metadata.

Relevant implementation:

- `st_pages/st_9_visualize_single_image.py`
- `_lookup_viewpoint(...)`
- `display_image(...)`

Risk under redesign:

The plan cuts this as a debug tool and suggests using the filesystem. That may be fine for engineers, but not for remote reviewers or nontechnical users who should not need shell access.

Recommendation:

Move this into Advanced/Scientist tools rather than deleting it. It can be small and low-priority.

## Usability Tradeoffs

### Gains Expected From the Redesign

- Fewer top-level pages.
- Less page-switching for field reviewers.
- Clear project status on startup.
- More obvious primary actions.
- Better review progress visibility.
- Less intimidating pipeline output.
- Cleaner default review interface.

### Usability Risks Introduced

- Power workflows may become hard to discover.
- Field staff may lose fast paths for high-confidence batches.
- Users may not understand project state if auto-load hides resume/restart behavior.
- Operators may lose access to raw errors needed to fix failed runs.
- Scientists may lose raw score and viewpoint evidence needed to tune thresholds.
- Not-matched review may become ambiguous if it is merged into the matched queue without special wording.
- Catalog update may become too hidden, even though it is required to complete a survey cycle.

## Proposed Redesign Requirements

Use these as acceptance criteria for the engineer implementing the redesign.

### Dashboard / Project Status

- Auto-load the current project if metadata exists.
- Show the loaded project path or name.
- Show last saved review timestamp if available.
- Show counts for total, reviewed, accepted known, assigned new ID, skipped, and remaining.
- Show pipeline step status for reference and query data separately where relevant.
- If query metadata does not exist, show a direct `Create Query Table` action.
- Include Settings/Admin for paths, thresholds, and advanced setup.
- Include guarded reset/restart controls; do not silently clear review inputs.

### Run Analysis

- Include detection/cropping, feature extraction, matching, unknown clustering, and catalog update.
- Support Reference Catalogue and Query Data where the underlying scripts require partition selection.
- Prevent steps from running when prerequisites are missing.
- Support per-step Run / Re-run and Run All.
- Support Stop current job.
- Parse progress into bars, but keep raw log access.
- Show actionable failure messages and missing prerequisite fixes.

### Review Matches

- Default to one query and one primary candidate.
- Preserve Accept, Reject/Assign New ID, and Skip.
- For below-threshold candidates, expose `Accept Candidate Anyway` or equivalent wording.
- Preserve candidate-rank cycling for all recommended IDs.
- Preserve browsing multiple reference images for the candidate individual.
- Preserve keypoint overlay as a collapsible panel.
- Preserve raw match evidence in an expandable panel.
- Preserve bulk actions in Advanced mode.
- Add filters for confidence, reviewed state, matched/not matched, skipped, and low-confidence cases.
- Add keyboard shortcuts only if they cannot accidentally fire while typing in inputs.
- Save review state clearly and show unsaved/saved status.

### Review Unknowns

- Show clusters as scrollable groups, as proposed.
- Keep all images in a cluster visible or quickly expandable.
- Support Confirm as new individual.
- Support Assign to known ID.
- Consider merge/split workflows if clusters are often imperfect.
- Preserve enough metadata per image to debug cluster quality: image ID, viewpoint, fused score if relevant, and source match information.
- Save confirmations in a way that remains compatible with downstream `assigned_individual_id` and `human_input` expectations.

### Scientist / Advanced Tools

- Keep ground-truth metric computation.
- Keep visual inspection of false positives, false negatives, and wrong known matches.
- Keep same-view/cross-view diagnostics.
- Keep single-image/viewpoint inspection.
- Keep raw metadata/result table access where safe.

## Implementation Notes

Several metadata fields are workflow contracts, not UI-only fields. The redesign should continue writing and reading them consistently:

- `human_input`
- `matching_status`
- `assigned_individual_id`
- `match_individual_{rank}`
- `match_image_{rank}`
- `match_fused_sim_{rank}`
- `match_global_sim_{rank}`
- `match_local_count_{rank}`
- `match_viewpoint_{rank}`
- `viewpoint`
- `viz_payload_{rank}` when available

Downstream partitioning, catalog update, and evaluation depend on these fields. Before changing button names or queue structure, map each new action to the exact metadata mutations it performs.

## Recommended Product Shape

The best version of the redesign is not three pages instead of ten; it is two operating modes over the same workflow:

- **Field Review Mode:** minimal dashboard, run status, review matches, review unknowns, big obvious actions, progress always visible.
- **Scientist / Operator Mode:** setup, bulk operations, raw evidence, logs, evaluation, error review, catalog update, and debug tools.

This keeps the plan's usability gains while preserving the capabilities that make the app useful for real survey operations and model improvement.