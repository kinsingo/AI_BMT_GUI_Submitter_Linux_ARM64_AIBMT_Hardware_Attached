#ifndef BMTSENSOR_H
#define BMTSENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ─────────────────────────────────────────────────────────────
// 결과 / 에러 코드
// ─────────────────────────────────────────────────────────────
typedef enum {
    SENSOR_OK                =  0,
    SENSOR_ERR               = -1,  // 일반 에러 (I2C 실패 등)
    SENSOR_ERR_INVALID_ARG   = -2,  // NULL 포인터, 잘못된 인자
    SENSOR_ERR_PARK_MISMATCH = -3,  // 파크된 채널이 아닌 다른 채널 읽기 시도
    SENSOR_ERR_NOT_PARKED    = -4   // sensor_get_park_mode: 현재 round-robin 상태
} SensorResult;

// ─────────────────────────────────────────────────────────────
// NPU 온도 센서 채널
// ─────────────────────────────────────────────────────────────
typedef enum {
    NPU_TOP = 0,
    NPU_BOTTOM = 1
} NpuChannel;

// ─────────────────────────────────────────────────────────────
// 전력 센서 채널
// ─────────────────────────────────────────────────────────────
typedef enum {
    POWER_NPU = 0,
    POWER_SYSTEM = 1,
    POWER_USB_A = 2,
    POWER_USB_C = 3
} PowerChannel;

// ─────────────────────────────────────────────────────────────
// 개별 채널 전력 데이터
// ─────────────────────────────────────────────────────────────
typedef struct {
    float voltage;  // 전압 (V)
    float current;  // 전류 (A)
    float power;    // 전력 (W)
} PowerChannelData;

// ─────────────────────────────────────────────────────────────
// 초기화 / 종료
//
// sensor_init:
//   - 매 호출마다 min_update_ms 를 기본값(1 ms)으로 리셋하고 파크 캐시를 해제.
//   - 이미 MAX34407 디바이스가 open된 상태에서 재호출되면 하드웨어 CONTROL
//     레지스터도 round-robin(0x00)으로 동기화하여 cache ↔ 하드웨어 desync 방지.
//   - 항상 SENSOR_OK 를 반환. 하드웨어 동기화 실패는 경고 로그만 출력.
//
// sensor_shutdown:
//   - 파크 모드가 활성이면 자동으로 해제(round-robin 복귀) 시도. 실패는 무시.
// ─────────────────────────────────────────────────────────────
int sensor_init(void);
int sensor_shutdown(void);

// ─────────────────────────────────────────────────────────────
// NPU 온도 읽기
//  - channel: NPU_TOP 또는 NPU_BOTTOM
//  - temp:    온도(°C) 출력 (성공 시에만 기록)
//  - return:  SensorResult (SENSOR_OK | SENSOR_ERR | SENSOR_ERR_INVALID_ARG)
// ─────────────────────────────────────────────────────────────
int sensor_read_npu_temp(NpuChannel channel, float* temp);

// ─────────────────────────────────────────────────────────────
// 전력 센서 읽기
//  sensor_read_voltage:
//    UPDATE(측정) 윈도우 없이 VOLTAGE 레지스터를 즉시 읽음.
//
//  sensor_read_current / sensor_read_power / sensor_read_power_channel:
//    update_ms 동안 UPDATE 윈도우로 누산 후 계산.
//    update_ms 가 최소값 미만(0 포함)이면 경고를 출력하고 자동으로 최소값으로
//    클램핑(예: update_ms=0 → 기본 최소 1 ms 로 측정).
//    최소값: 기본 1 msec, sensor_set_min_update_ms 로 변경 가능.
//
//  반환값: SensorResult
//    SENSOR_OK                : 성공, 출력 포인터에 값 기록
//    SENSOR_ERR_INVALID_ARG   : NULL 포인터
//    SENSOR_ERR_PARK_MISMATCH : 현재 파크된 채널과 다른 채널 요청
//    SENSOR_ERR               : I2C 등 일반 에러
// ─────────────────────────────────────────────────────────────
int sensor_read_voltage(PowerChannel channel, float* voltage);
int sensor_read_current(PowerChannel channel, uint32_t update_ms, float* current);
int sensor_read_power  (PowerChannel channel, uint32_t update_ms, float* power);
int sensor_read_power_channel(PowerChannel channel, PowerChannelData* data, uint32_t update_ms);

// ─────────────────────────────────────────────────────────────
// 채널 파크 모드 (Channel Park Mode)
//  MAX34407 CONTROL.PARK_EN 을 설정하여 하나의 채널만 4배 빠르게 샘플링.
//  파크 모드 활성 중 다른 채널을 읽으려고 하면 SENSOR_ERR_PARK_MISMATCH 를 반환.
// ─────────────────────────────────────────────────────────────
int sensor_set_park_mode(PowerChannel channel);  // 파크 활성
int sensor_clear_park_mode(void);                // 파크 해제 → round-robin 복귀

// sensor_get_park_mode:
//   파크 on  → 해당 PowerChannel 값 (0~3)
//   파크 off → SENSOR_ERR_NOT_PARKED (-4)
int sensor_get_park_mode(void);

// ─────────────────────────────────────────────────────────────
// 최소 UPDATE 시간 설정 (msec)
//  - 기본값: 1 msec
//  - set: ms == 0 이면 SENSOR_ERR_INVALID_ARG
// ─────────────────────────────────────────────────────────────
int      sensor_set_min_update_ms(uint32_t ms);
uint32_t sensor_get_min_update_ms(void);


#ifdef __cplusplus
}
#endif

#endif // BMTSENSOR_H
