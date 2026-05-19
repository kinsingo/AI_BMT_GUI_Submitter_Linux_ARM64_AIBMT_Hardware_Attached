#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include "bmtsensor.h"

using namespace std;
using namespace cv;

class Classification_Implementation_DXNN : public AI_BMT_Interface
{
    shared_ptr<dxrt::InferenceEngine> ie;
    int resolution;

public:
    Classification_Implementation_DXNN(int resolution = 224) : resolution(resolution)
    {
        cout << "resolution : " << this->resolution<<endl;
        if (sensor_init() != SENSOR_OK) {
            fprintf(stderr, "센서 초기화 실패\n");
        }
    }

    virtual ~Classification_Implementation_DXNN()
    {
        sensor_shutdown();
    }

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
        data.accelerator_type = "DeepX M1";
        data.submitter = "DX-RT(v3.3.0) DX-COM(v2.3.0) FW(v2.5.6) Driver(v2.4.1)";
        data.cooling_option = to_string(this->resolution);
        data.operating_system = "Ubuntu22.04 LTS"; // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual void initialize(string modelPath) override
    {
        cout << "Initialze() is called .." << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        cout << "Initialze() is finished .. " << endl;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat input;
        input = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        if (resolution != 224) {
            cv::resize(input, input, cv::Size(resolution, resolution));
        }
        vector<uint8_t> inputBuf(ie->GetInputSize(), 0);
        memcpy(&inputBuf[0], &input.data[0], ie->GetInputSize());
        return inputBuf;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> queryResult;
        const int querySize = data.size();
        for (int i = 0; i < querySize; i++)
        {
            vector<uint8_t> inputBuf = get<vector<uint8_t>>(data[i]);
            BMTVisionResult result;
            auto outputs = ie->Run(inputBuf.data());
            float *output_data = (float *)outputs.front()->data();
            vector<float> output(output_data, output_data + 1000);
            result.classProbabilities = output;
            queryResult.push_back(result);
        }
        return queryResult;
    }
};