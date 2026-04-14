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
    int align_factor;
    int input_w = 224, input_h = 224, input_c = 3;
    bool isCustomDataset;

public:
    Classification_Implementation_DXNN(bool isCustomDataset)
    {
        this->isCustomDataset = isCustomDataset;
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
        if (isCustomDataset)
            return InterfaceType::ImageClassification_CustomDataset;
        else
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
        cout << "Initialze() is called .." << endl;
        ie = make_shared<dxrt::InferenceEngine>(modelPath);
        cout << "Initialze() is finished .. " << endl;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat input;
        input = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        if (isCustomDataset)
        {
            const int target_short = 232;
            const int crop = 224;

            int h = input.rows;
            int w = input.cols;

            // 1) 짧은 변을 232로 맞추는 비율 (종횡비 유지)
            double scale = static_cast<double>(target_short) / std::min(h, w);
            int new_w = static_cast<int>(std::round(w * scale));
            int new_h = static_cast<int>(std::round(h * scale));

            // Downscale면 INTER_AREA, Upscale면 INTER_LINEAR 권장
            int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

            cv::Mat resized;
            cv::resize(input, resized, cv::Size(new_w, new_h), 0, 0, interp);

            // 2) 중심 224x224 크롭
            int x = (resized.cols - crop) / 2;
            int y = (resized.rows - crop) / 2;
            cv::Rect roi(x, y, crop, crop);
            input = resized(roi).clone();
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