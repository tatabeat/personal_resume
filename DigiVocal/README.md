# DigiVocal｜數位發聲系統（研究原型）

DigiVocal 是一個探索**可控語音生成**的本機研究原型。它將中文文字轉換為帶聲調的拼音，建立構音／發聲控制時間軸，再以可調的聲源與濾波流程產生 WAV。目標是逐步探索音高、音色、韻律與情緒表現的控制，不是現成的高品質 TTS 服務。

> **目前狀態：**能產生具有音高起伏的初步發聲，並檢視與調整波形；音節清晰度及整句可理解度仍待改善。

## 目前能做什麼

- 中文文字 → 拼音／聲調與基本變調 → 發聲動作時間軸 → 波形合成。
- 在 Tkinter GUI 調整基頻、共振峰、呼吸感、波形與其他參數，播放或匯出 WAV。
- 匯出動作 JSON、供後續研究使用的 VocalTractLab 橋接 JSON；產生音訊診斷報告。
- 載入或儲存聲音輪廓 JSON；`profiles/digivocal_voice_profile.json` 是隨專案提供的研究設定，**不會在啟動時自動載入**，可在 GUI 中手動選擇。

## 快速開始（Windows）

1. 安裝 Python 3.12，並確認已安裝 Python Launcher（在命令提示字元執行 `py -3.12 --version`）。
2. 下載或 clone 本專案，在專案根目錄執行 `install_windows.bat`。此腳本會建立 `.venv` 並安裝本專案，不會將套件安裝到全域 Python。
3. 執行 `run_windows.bat`。在文字欄輸入中文，先按「分析動作」，再按「合成並播放」或「只合成」。
4. 如需使用隨附聲線設定，在介面選擇「載入聲音輪廓 JSON」，開啟 `profiles/digivocal_voice_profile.json`。

也可以在 Python 3.10–3.13 的其他環境建立自己的虛擬環境，執行 `python -m pip install -e .`，然後以 `python main.py` 或 `digivocal-gui` 啟動；GUI 需要可用的 Tkinter/桌面環境，播放功能需要音訊裝置。Windows 批次檔固定使用 Python 3.12。

## 專案結構

```text
main.py                 GUI 啟動入口
src/digivocal/          拼音前端、動作規劃、合成器、GUI 與診斷模組
profiles/               研究用聲音輪廓 JSON
tests/                  不依賴音效裝置的基本測試
pyproject.toml          套件資訊與依賴
requirements.txt        基礎依賴清單（方便直接安裝）
```

可選後端（例如 g2pW、SPARC、CuPy）並非基本安裝所必需；`pyproject.toml` 的 optional dependencies 列出對應額外依賴。VocalTractLab 橋接 JSON 只是中介資料，並非可直接使用的 `.ges` 檔。

## 測試與限制

安裝後，在專案根目錄執行：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

測試檢查拼音 token、聲音輪廓載入及短音訊合成；它不是主觀聽感或咬字品質評測。合成診斷會在 `Debug_log/` 產生檔案，已從版本控制排除。使用 GUI 匯出的 WAV／JSON 請自行選擇保存位置，不要意外把私人資料一起提交。

## 公開前請確認

- 隨附的 `profiles/digivocal_voice_profile.json` 與 `src/digivocal/voice_profile.py` 包含聲源／參考波形資料；公開前請確認原始錄音、聲線與相關素材的權利和當事人同意範圍。
- 此整理版**尚未指定開源授權**。請由專案權利人與共同作者確認後，才加入 `LICENSE` 並在 `pyproject.toml` 宣告授權。公開 GitHub repository 不等於自動採用 MIT。
- 若要對外展示，建議額外提供介面截圖及獲准公開的短音訊樣本，並標註產生方法與現階段限制。
