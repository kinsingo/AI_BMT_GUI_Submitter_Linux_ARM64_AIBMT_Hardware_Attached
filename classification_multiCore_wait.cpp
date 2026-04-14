#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include <getopt.h>
#include <future>
#include <thread>
#include <iostream>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "bmtsensor.h"

using namespace std;
using namespace cv;

class Classification_Implementation_DXNN_MultiThreads : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int input_w = 224, input_h = 224, input_c = 3;
    const int maxConcurrentRequests = 64;

public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    virtual PowerDeviceType getPowerDeviceType() override
    {
        return PowerDeviceType::CustomDevice;
    }

    virtual vector<CustomPowerSample> measureCustomPower() override
    {
        const char* channel_names[] = {"POWER_NPU", "POWER_SYSTEM", "POWER_USB_A", "POWER_USB_C"};
        vector<CustomPowerSample> samples;
        for (int i = 0; i < 4; i++) {
            PowerChannel ch = (PowerChannel)i;
            PowerChannelData data;
            if (sensor_read_power_channel(ch, &data, 2) == SENSOR_OK) {
                samples.push_back({channel_names[i], data.power});
            } else {
                samples.push_back({channel_names[i], 0.0});
            }
        }
        return samples;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "AI-BMT-Hardware";
        data.accelerator_type = "M1(NPU)";
        data.submitter = "DeepX";
        data.operating_system = "Ubuntu22.04 LTS"; // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        cout << "Initialze() is called" << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        cout << "maxConcurrentRequests : " << maxConcurrentRequests << endl;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat input;
        input = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        vector<uint8_t> inputBuf(ie->GetInputSize(), 0);
        memcpy(&inputBuf[0], &input.data[0], ie->GetInputSize());
        return inputBuf;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        int querySize = data.size();
        vector<BMTVisionResult> queryResult(querySize);
        vector<int> reqIds(querySize);
        vector<vector<uint8_t>> inputBufs(querySize);

        for (int i = 0; i < querySize; i += maxConcurrentRequests)
        {
            int currentBatchSize = min(maxConcurrentRequests, querySize - i);

            // Start a batch of RunAsync calls
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                inputBufs[index] = get<vector<uint8_t>>(data[index]);
                reqIds[index] = (ie->RunAsync(inputBufs[index].data()));
            }

            // 병렬 처리로 Wait 실행
            vector<std::future<void>> futures;
            for (int j = 0; j < currentBatchSize; j++)
            {
                int index = i + j;
                futures.push_back(std::async(std::launch::async, [&, index]()
                                             {
                auto outputs = ie->Wait(reqIds[index]);
                inputBufs[index].clear(); // input buffer 해제
                BMTVisionResult result;
                float *output_data = (float *)outputs.front()->data();
                vector<float> output(output_data, output_data + 1000);
                result.classProbabilities = output;
                queryResult[index] = result; }));
            }

            // 모든 요청 완료 대기
            for (auto &f : futures)
            {
                f.get();
            }
        }
        return queryResult;
    }
};
