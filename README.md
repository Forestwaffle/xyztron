## 동작 흐름

```text
프로그램 시작
  ↓
카메라 구독 시작
라이다 구독 시작
모터 Publisher 준비
  ↓
mission_state = WAIT_TRAFFIC_LIGHT
  ↓
카메라 이미지 수신 대기
  ↓
신호등 감지
  ↓
빨간불 또는 UNKNOWN
  → 정지 유지
  ↓
초록불 감지
  ↓
신호등 감지 종료
디버그 창 닫기
  ↓
mission_state = CONE_DRIVE
  ↓
라이다 값 확인
  ↓
라이다 없음
  → 정지 대기
  ↓
라이다 있음
  ↓
ConeLidarDriver 시작
  ↓
GO 상태
  ↓
left/right 중앙값 계산
  ↓
left 또는 right < 10m ?
  ├─ 아니오 → 직진
  └─ 예 → TURN
              ↓
          왼쪽 최대 조향
              ↓
          left > 53m 또는 left invalid ?
              ├─ 아니오 → 계속 왼쪽 최대 조향
              └─ 예 → FORWARD
                          ↓
                      직진 유지
                          ↓
                      mission_state = AUTO_DRIVE



