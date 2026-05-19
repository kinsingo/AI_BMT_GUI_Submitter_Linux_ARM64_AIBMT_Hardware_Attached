#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "dxrt/dxrt_api.h"
#include "classification_multiCore_wait.cpp"
#include "classification_singleCore.cpp"
#include "bmtsensor.h"
using namespace std;

int main(int argc, char *argv[])
{
   /*
    printf("=== BMT 센서 C++ 샘플 프로그램 ===\n\n");
    
    printf("최소 UPDATE 시간: %u ms (기본값)\n", sensor_get_min_update_ms());
    sensor_set_min_update_ms(5);
    printf("최소 UPDATE 시간 변경 후: %u ms\n\n", sensor_get_min_update_ms());

    // 센서 초기화
    if (sensor_init() != SENSOR_OK) {
        fprintf(stderr, "센서 초기화 실패\n");
        return 1;
    }

    printf("\n");

    // 개별 채널 온도 읽기
    float npu_top = 0.0f, npu_bottom = 0.0f;
    int r1 = sensor_read_npu_temp(NPU_TOP, &npu_top);
    int r2 = sensor_read_npu_temp(NPU_BOTTOM, &npu_bottom);

    printf("개별 채널 온도:\n");
    printf("  NPU_TOP    (0x48): %.4f °C  [%s]\n", npu_top,    r1 == SENSOR_OK ? "OK" : "ERR");
    printf("  NPU_BOTTOM (0x49): %.4f °C  [%s]\n\n", npu_bottom, r2 == SENSOR_OK ? "OK" : "ERR");

    // 개별 채널 전압 읽기 (즉시)
    printf("개별 채널 전압 읽기:\n");
    const char* channel_names[] = {"POWER_NPU", "POWER_SYSTEM", "POWER_USB_A", "POWER_USB_C"};
    for (int i = 0; i < 4; i++) {
        PowerChannel ch = (PowerChannel)i;
        float voltage = 0.0f;
        sensor_read_voltage(ch, &voltage);
        printf("  %s: V=%.6f V\n", channel_names[i], voltage);
    }
    printf("\n");

    // 개별 채널 전력 데이터 읽기 (한 번에)
    printf("개별 채널 전력 데이터 읽기 (각 100ms UPDATE, round-robin):\n");
    for (int i = 0; i < 4; i++) {
        PowerChannel ch = (PowerChannel)i;
        PowerChannelData data = {0};
        if (sensor_read_power_channel(ch, &data, 100) == SENSOR_OK) {
            printf("  %s: V=%.6f V  I=%.6f A  P=%.6f W\n",
                   channel_names[i], data.voltage, data.current, data.power);
        } else {
            printf("  %s: 측정 실패\n", channel_names[i]);
        }
    }
    printf("\n");

    // 센서 종료
    sensor_shutdown();
    */

/*
aibmtExample@gmail.com

rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
     cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
cmake -G "Ninja" ..
cmake --build .
xauth extract /tmp/.Xauth_bmt $DISPLAY
sudo DISPLAY=$DISPLAY XAUTHORITY=/tmp/.Xauth_bmt LD_LIBRARY_PATH=$(pwd)/lib ./AI_BMT_GUI_Submitter
*/

    try
    {
       //bool resolution = 224;
       //shared_ptr<AI_BMT_Interface> interface = make_shared<Classification_Implementation_DXNN>();
       shared_ptr<AI_BMT_Interface> interface = make_shared<Classification_Implementation_DXNN_MultiThreads>();
       return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
       cout << ex.what() << endl;
    }
}
