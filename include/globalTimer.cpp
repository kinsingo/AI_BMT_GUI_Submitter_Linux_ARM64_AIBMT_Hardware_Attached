#include "globalTimer.h"

chrono::high_resolution_clock::time_point GlobalTimer::start;
chrono::high_resolution_clock::time_point GlobalTimer::end;
bool GlobalTimer::isMeasured = false;
bool GlobalTimer::started = false;
bool GlobalTimer::isPythonAPIMeasure = false;

// Python 여부는 외부에서 설정 (default: false)
bool GlobalTimer::IsPythonAPIMeasure()
{
    return isPythonAPIMeasure;
}

void GlobalTimer::SetIsPythonAPIMeasure(bool isPythonAPI)
{
    isPythonAPIMeasure = isPythonAPI;
}

void GlobalTimer::onInferenceStart() {
    // [Validation first] Check validity before time measurement to minimize timing distortion
    if (started)
        throw runtime_error("[GlobalTimer::onInferenceStart()] Start already called without calling End. Cannot start again.");
    started = true;
    isMeasured = false;

    // [Timing last] Perform time measurement as the last operation to avoid measuring overhead of validation
    start = chrono::high_resolution_clock::now();
}

void GlobalTimer::onInferenceEnd() {
    // [Timing first] Capture end time as early as possible to exclude validation overhead
    end = chrono::high_resolution_clock::now();

    // [Validation after] Perform validation after measurement to avoid affecting latency results
    if (!started)
        throw runtime_error("[GlobalTimer::onInferenceEnd()] Start must be called before End.");
    if (isMeasured)
        throw runtime_error("[GlobalTimer::onInferenceEnd()] Measurement already completed. Call getMeasuredLatencyMs() before measuring again.");
    isMeasured = true;
    started = false;
}

//This function does not affect time measurement
double GlobalTimer::getMeasuredLatencyMs() {
    if (!isMeasured)
        throw runtime_error("[GlobalTimer::getMeasuredLatencyMs()] Measurement not available. Did you forget to call onInferenceEnd()?");
    isMeasured = false;
    return chrono::duration<double, milli>(end - start).count();
}
