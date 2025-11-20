#include "hailo/hailort.hpp"
#include "ai_bmt_interface.h"
#include "ai_bmt_gui_caller.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <map>

using namespace hailort;
using namespace std;

class HailoBMT : public AI_BMT_Interface
{
private:
    std::unique_ptr<VDevice> m_vdevice;
    std::shared_ptr<ConfiguredNetworkGroup> m_network_group;
    std::unique_ptr<InferVStreams> m_pipeline;
    std::string m_input_name;
    std::string m_output_name;
    bool isCustomDataset;

    // 🔧 이전 리소스를 정리
    void cleanup()
    {
        // 1) VStreams 파이프라인 파괴
        if (m_pipeline)
        {
            m_pipeline.reset(); // dtor가 스트림 닫음
        }

        // 2) NetworkGroup 종료 (스케줄러/레벨에서 스트림/컨텍스트 닫기)
        if (m_network_group)
        {
            (void)m_network_group->shutdown(); // 상태 반환 무시 (이미 종료 중일 수도 있음)
            m_network_group.reset();
        }

        // 3) VDevice 해제 (물리 디바이스 핸들 반납)
        if (m_vdevice)
        {
            m_vdevice.reset();
        }
    }

public:
    HailoBMT(bool isCustomDataset)
    {
        this->isCustomDataset = isCustomDataset;
    }

    InterfaceType getInterfaceType() override
    {
        if (isCustomDataset)
            return InterfaceType::ImageClassification_CustomDataset;
        else
            return InterfaceType::ImageClassification;
    }

    Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Broadcom BCM2712 quad-core Arm Cortex A76 @ 2.4GHz";
        data.accelerator_type = "Hailo-8";
        data.submitter = "Hailo";
        data.cpu_core_count = "4";
        data.cpu_ram_capacity = "8GB";
        data.cooling = "Air";
        data.cooling_option = "Active";
        data.cpu_accelerator_interconnect_interface = "PCIe 3.0 x4";
        data.benchmark_model = "mobilenet_v2_opset10";
        data.operating_system = "Ubuntu 24.04.2 LTS";
        return data;
    }

    void initialize(string modelPath) override
    {
        cleanup();

        m_vdevice = VDevice::create().release();

        auto hef = Hef::create(modelPath).value();
        auto config_params = m_vdevice->create_configure_params(hef).value();
        auto network_groups = m_vdevice->configure(hef, config_params).value();
        m_network_group = network_groups[0];

        auto input_params = m_network_group->make_input_vstream_params(
                                               false, HAILO_FORMAT_TYPE_UINT8,
                                               HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                               HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
                                               "")
                                .value();

        auto output_params = m_network_group->make_output_vstream_params(
                                                false, HAILO_FORMAT_TYPE_FLOAT32,
                                                HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                HAILO_DEFAULT_VSTREAM_QUEUE_SIZE,
                                                "")
                                 .value();

        auto infer_expected = InferVStreams::create(*m_network_group, input_params, output_params);
        if (!infer_expected)
        {
            throw std::runtime_error("InferVStreams::create failed");
        }
        m_pipeline = std::make_unique<InferVStreams>(std::move(*infer_expected));

        m_input_name = input_params.begin()->first;
        m_output_name = output_params.begin()->first;
    }

    VariantType preprocessVisionData(const string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        if (isCustomDataset)
        {
            const int target_short = 232;
            const int crop = 224;

            int h = img.rows;
            int w = img.cols;

            // 1) 짧은 변을 232로 맞추는 비율 (종횡비 유지)
            double scale = static_cast<double>(target_short) / std::min(h, w);
            int new_w = static_cast<int>(std::round(w * scale));
            int new_h = static_cast<int>(std::round(h * scale));

            // Downscale면 INTER_AREA, Upscale면 INTER_LINEAR 권장
            int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, interp);

            // 2) 중심 224x224 크롭
            int x = (resized.cols - crop) / 2;
            int y = (resized.rows - crop) / 2;
            cv::Rect roi(x, y, crop, crop);
            img = resized(roi).clone();
        }

        std::vector<uint8_t> input_buf(img.total() * img.channels());
        std::memcpy(input_buf.data(), img.data, input_buf.size());
        return input_buf;
    }

    std::vector<BMTVisionResult> inferVision(const std::vector<VariantType> &data) override
    {
        std::vector<BMTVisionResult> results;
        results.reserve(data.size());

        for (const auto &item : data)
        {
            const auto &input_buf = std::get<std::vector<uint8_t>>(item);
            std::map<std::string, MemoryView> input_data;
            std::map<std::string, MemoryView> output_data;
            input_data[m_input_name] = MemoryView(const_cast<uint8_t *>(input_buf.data()), input_buf.size());
            std::vector<float> output_buf(1000);
            output_data[m_output_name] = MemoryView(output_buf.data(), output_buf.size() * sizeof(float));
            m_pipeline->infer(input_data, output_data, /*batch_size*/ 1);

            BMTVisionResult r;
            r.classProbabilities = std::move(output_buf);
            results.push_back(std::move(r));
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        bool isCustomDataset = false;
        std::shared_ptr<AI_BMT_Interface> interface = std::make_shared<HailoBMT>(isCustomDataset);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
        return -1;
    }
}
