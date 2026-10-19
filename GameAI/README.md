# GameAI｜雙手遊戲畫面辨識與操作原型

這是以 Python GUI 組成的**遊戲 AI 巨集／研究原型**：擷取指定的遊戲畫面，讓左右手（L/R）各自經過 `mind`、`decide`、`quick` 階段，再把文字指令排入 `action` 佇列，以滑鼠操作。三組設定範例位於 `dataset/AOV`、`dataset/BoomBoomGame` 和 `dataset/Snake`。這不是已能通用地玩任何遊戲的完成品；畫面座標、提示詞和動作規則仍依場景調整。

## 專案構造

| 部分 | 用途 |
| --- | --- |
| `main.py`、`gui.py` | 啟動桌面介面，框選畫面、選擇設定資料夾、控制執行與停止 |
| `catch.py`、`scshot.py` | 畫面範圍預覽與截圖 |
| `gpt.py`、`dataset/` | OpenAI Responses API 呼叫、各遊戲／左右手階段設定 |
| `mind.py`、`decide.py`、`quick.py` | 各階段思考與操作指令生成 |
| `action.py` | 操作佇列、指令解析及滑鼠執行 |
| `log.py` | GUI 訊息與本機截圖紀錄 |

## Windows 啟動

1. 安裝 Python 3.12，執行 `install_windows.bat`，套件會安裝到專案內的 `.venv`。
2. 在 PowerShell 進入專案目錄，以不回顯輸入方式設定**這次工作階段**的金鑰：

   ```powershell
   $secureKey = Read-Host "OpenAI API key" -AsSecureString
   $keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
   try { $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr) }
   finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr) }
   .\run_windows.bat
   Remove-Item Env:\OPENAI_API_KEY
   ```

   也可先在 Windows 的使用者環境變數設定 `OPENAI_API_KEY`，再重新開啟終端機。不要寫進原始碼、`key.json` 或批次檔。**原壓縮檔內的金鑰未放入本版；若那是可用的金鑰，建議撤銷並產生新金鑰。**
3. 在 GUI 選擇 `dataset` 底下的遊戲設定資料夾，按 `Select Area` 指定遊戲視窗區域；先檢查區域，再按 `Start`。結束時按 `Stop`。

設定檔的 `enabled` 控制各階段是否運行，`model` 需改為你的帳戶實際可用的模型。API 請求可能產生費用；截圖會發送到外部模型服務，請勿框選含有私人資訊的桌面區域。`logs/` 會儲存畫面及模型輸出，已從 Git 排除；使用者若手動上傳紀錄仍可能洩漏個資。

## 指令與目前限制

操作輸出使用 `c(x,y)`（點擊）、`p(x,y)`（按住並移動）、`s(x1,y1)(x2,y2)`（拖曳），以及 `e`、`n`（不操作）。座標相對於所框選的畫面左上角。左右手佇列互相參照，但仍需逐遊戲校正提示詞、區域與反應時間。

**注意：**原型使用 `PyAutoGUI` 控制真實滑鼠，且 `action.py` 將 `FAILSAFE` 設為 `False`。第一次執行請在不會造成損失的測試畫面檢查操作，必要時從 GUI 停止或結束程式。`mouse/MouseDll.dll` 和其獨立示例沒有被主程式使用，且缺少可供審核的 DLL 原始碼，因此不包含在此公開版。

## 驗證與發布

`python -m unittest discover -s tests -v` 可執行不需要桌面或有效 API 金鑰的基本測試。本版尚未指定開源授權；請確認程式、遊戲素材與提示詞的公開權利後，再由權利人決定是否加入 `LICENSE`。本機 Windows GUI、真實遊戲畫面與 API 連線仍須在你的環境中驗證。
