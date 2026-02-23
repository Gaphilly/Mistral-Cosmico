# Task Proposals from Codebase Review

## 1) Typo fix task
**Task:** Fix the UI label typo from `Frecuencia lluvia` to `Frecuencia de lluvia` in the sidebar.

**Why:** The label is missing the preposition `de`, which reads as a typo/awkward Spanish phrasing.

**Acceptance criteria:**
- The default sidebar label reads `🌧️ Frecuencia de lluvia:`.
- The dynamic JavaScript fallback label also uses `Frecuencia de lluvia (>2mm)` for consistency.

## 2) Bug fix task
**Task:** Align backend wind metric key with frontend expectation.

**Why:** The frontend reads `strong_winds_freq_percent`, but the backend returns `avg_10m_wind_speed_m_s`; this mismatch causes the wind progress bar to stay at `0%`.

**Acceptance criteria:**
- The `/climate_stats` response includes `strong_winds_freq_percent`.
- Frontend wind bar updates using actual backend data.
- Add a route-level test for key presence and type.

## 3) Documentation/comment discrepancy task
**Task:** Correct outdated comments in `compute_wind_speed_stats` to match actual behavior.

**Why:** The function currently computes yearly maximum wind events (`daily_max = np.max(...)`) and then averages those yearly booleans, but comments claim daily aggregation and “frequency percent over 15 years.” This is misleading and can cause incorrect maintenance assumptions.

**Acceptance criteria:**
- Comments explicitly describe that the current implementation computes a yearly occurrence metric.
- README “Detailed results” section is updated to describe wind output consistently with implementation.
- If desired output is actually daily frequency, open a follow-up implementation task.

## 4) Test improvement task
**Task:** Add unit tests for request validation and precipitation label branching in `/climate_stats`.

**Why:** Critical behavior currently lacks tests:
- Missing query params should return 400.
- Precipitation key should switch between `snow_hail_freq_percent` and `rainfall_gt_2mm_freq_percent` based on temperature threshold.

**Acceptance criteria:**
- Test: missing `day/month/lat/lon` returns 400 with clear JSON error.
- Test: mocked cold average temp returns `snow_hail_freq_percent`.
- Test: mocked mild temp returns `rainfall_gt_2mm_freq_percent`.
- Tests run in CI without network calls (mock external functions).
