POLICY-CONDITIONED CONFIDENCE: COMPUTATIONAL REFERENCE
====================================================

Code version: 0.1.0
Copyright (c) 2026 Changhyun Jun. MIT License; see LICENSE.

This package accompanies a controlled exploratory study of policy-conditioned
confidence in a frozen vision-language model. It is computation-only reference
code, not independent reproduction of private-data study results. No private
media, per-item human observations/labels, model responses/predictions, feature
arrays or trained checkpoints are included. Synthetic tests are not study data.

Run synthetic code checks
-------------------------
Use Python 3.11 and requirements-test.txt. Provide a locally installed
DejaVuSans.ttf via the environment variable POLICY_TEST_FONT, then run:

    python run_synthetic_tests.py --report SYNTHETIC_TEST_REPORT.json

Only fictitious records and synthetic tensors/images are used. No dataset is
loaded, no VLM/checkpoint is loaded, no optimizer is created, and no parameters
are trained. The font is an external dependency and is not bundled. The recorded
font digest is in ROI_RENDERING_CONFIG.json. Renderer tests explicitly skip if
no test font is supplied; inspect skips before reporting a complete test pass.

Included code
-------------
models.py, _reference_ranking.py, _reference_metrics.py, _reference_cache.py:
  Original confidence heads, ranking/exposure computations, metrics and cache
  loader. Their copied function bodies are unchanged.
api.py:
  Existing guarded adapter for input/ID checks, priors, q(actual generated
  decision), null-aware C checks, pair comparisons and matched rewiring.
linear.py, calibration.py:
  Original numerical primitives and diagnostics. No fitting runner is included.
response_parser.py, vlm_interface.py:
  Original parser, separate strict-format parser, image RGB conversion and
  official chat-template/token-interface check, candidate softmax, and extraction
  from an already supplied generation-step-zero hidden-state object. These do
  not load a model or generate an answer. The historical parser is permissive;
  parse success is not evidence of strict-format compliance.
rule_mapper.py, evidence_gate.py:
  Original priority-ordered rule logic and evidence gate. An unresolved raw rule
  output is retained separately from a null evaluation label. This is not an
  independently validated ground-truth system. Use resolve/derive_all_gated,
  not evaluate alone, when producing evaluation labels.
roi.py:
  Temporal PNG drawing operations with only configuration/font paths exposed as
  arguments. Pass an RGB image, clip['roi_bounds_inclusive_xyxy'], the supplied
  rendering configuration and a font path. Caller supplies already fixed spatial
  bounds; this code performs no detection, tracking, cropping or resizing.
  Its historical coordinate assumptions are retained. It is not a generic ROI
  editor and does not reconstruct the earlier JPG annotation/encoding pipeline.

Configuration and provenance
----------------------------
REFERENCE_CONFIG.json retains fixed model, feature, optimizer, prior and metric
definitions. fixed_boundary_policies.json contains permitted verbatim policies.
POLICY_RULE_TABLE.json removes private source paths/item identifiers, not policy
clauses, priorities or logical rules. FUNCTION_PROVENANCE.json records unchanged
function segment digests and explicit adapters. Internal provenance paths are
not distributed. CITATION.cff intentionally omits unavailable DOI, repository,
release date rather than inventing them; the license is MIT.

Scope and limitations
---------------------
This is a computational reference, NOT independent reproduction of the paper's
results. No private media, human observations, per-item predictions, hidden
caches, trained checkpoints, train/held-out IDs or source-group records are
included. Whole-study OOF/transfer orchestration, optimization runners, full VLM
generation and later prompt-only-forward runners, and observation UIs remain
outside this candidate. Primitive tests do not establish those workflows.

Generation-step-zero hidden/logit extraction and later prompt-forward scores
have distinct provenance. Missing historical generation logits cannot be
recovered or declared bit-identical by this package. Historical JPG processing
is not fully recovered. The temporal PNG renderer is a later, separate version.

Verification and citation
-------------------------
SYNTHETIC_TEST_REPORT.json is the preserved 47-test report from the tested code
candidate. Its code SHA-256 values match every Python file in this package.
Packaging and metadata preparation did not rerun tests or perform experiments.
PACKAGE_MANIFEST.json lists all release files except itself. The archive carries
no prior Git history, automation credentials or embedded private data.

CITATION.cff provides the author, affiliation, software title, version and MIT
license. A DOI and release date will be added only after actually issued; none
is invented. Third-party libraries and fonts are not bundled and retain their
own terms; see THIRD_PARTY_NOTICES.txt.

The study's finding remains NOT_ESTABLISHED_IN_CURRENT_SETUP. This package does
not establish an advantage or absence of benefit of policy-matched ranking.
