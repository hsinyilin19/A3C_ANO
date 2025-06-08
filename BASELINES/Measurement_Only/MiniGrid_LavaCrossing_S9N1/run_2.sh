#!/bin/bash

LOGDIR="logs_qubit_4_depth_2_4_6_rotation_only"

# run_and_log() {
#   SCRIPT_NAME=$1
#   SCRIPT_BASE="${SCRIPT_NAME%.*}"  # 去除副檔名 .py
#   LOG_FILE="$LOGDIR/${SCRIPT_BASE}_$(date +%Y%m%d_%H%M%S).log"

#   # 建立 log 檔案需要的目錄（包含 subfolder）
#   mkdir -p "$(dirname "$LOG_FILE")"

#   echo "🔥 啟動 $SCRIPT_NAME - 輸出儲存於 $LOG_FILE"
#   echo "========== $SCRIPT_NAME 開始於 $(date) ==========" >> "$LOG_FILE"
#   /usr/bin/time -v python "$SCRIPT_NAME" >> "$LOG_FILE" 2>&1
#   echo "✅ $SCRIPT_NAME 結束於 $(date)" >> "$LOG_FILE"
#   echo
# }

run_and_log() {
  SCRIPT_NAME=$1
  SCRIPT_BASE="${SCRIPT_NAME%.*}"
  LOG_FILE="$LOGDIR/${SCRIPT_BASE}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$(dirname "$LOG_FILE")"

  echo "🔥 啟動 $SCRIPT_NAME - 輸出儲存於 $LOG_FILE"
  echo "========== $SCRIPT_NAME 開始於 $(date) ==========" | tee -a "$LOG_FILE"

  # 強制 Python 線性 flush，並用 tee 同步到畫面和檔案
  PYTHONUNBUFFERED=1 /usr/bin/time -v python "$SCRIPT_NAME" 2>&1 | tee -a "$LOG_FILE"

  echo "✅ $SCRIPT_NAME 結束於 $(date)" | tee -a "$LOG_FILE"
  echo
}

# 🔁 強制序列啟動
run_and_log "qubit_4_depth_2_n_local_3_100k/discrete_A3C_vqc_nonlocal_v2_Classical_Encoder_Measurement_Only.py"
run_and_log "qubit_4_depth_4_n_local_3_100k/discrete_A3C_vqc_nonlocal_v2_Classical_Encoder_Measurement_Only.py"
run_and_log "qubit_4_depth_6_n_local_3_100k/discrete_A3C_vqc_nonlocal_v2_Classical_Encoder_Measurement_Only.py"


echo "🚀 全部程式執行完畢。Mini PC 應該已經冒煙（或變得更強）了！"