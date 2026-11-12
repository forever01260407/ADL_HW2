# 最後一步：提供最終模型的下載連結

正式提交學號：**r15922a14**。

已準備、應上傳的模型檔：

```text
output/models/h5_final_r15922a14.zip
```

- 大小：395,347,153 bytes（約395 MB）。
- SHA-256：`714d7a9be5486de5db8c59a8fba1a9f65d4e868f3c00b668500c20e5d8309768`。
- 內含已通過本機 Public Boss 的原H5 checkpoint，不是W11另存的重訓輸出。
- 不必把整個專案、Public資料或開發環境一起上傳到模型託管位置。

把這個檔案放到你已有的託管位置，提供助教無需互動登入即可取得的HTTPS下載連結。一般「分享預覽頁」可能回傳HTML，需要轉成可下載檔案的連結；下載工具會驗證SHA，不會把HTML當成權重。

提供連結後，尚需完成以下最後驗收：

1. 將連結寫入`code/script/configs/release_assets.json`的`final_model.url`。
2. 在沒有模型快取的目錄實際下載一次，核對archive及全部checkpoint檔案SHA；不重跑Public Judge。
3. 重新封裝成`output/r15922a14.zip`，解壓後最外層為`r15922a14/`。
4. 上傳NTU COOL前自行確認檔案，下載連結需在截止後至少維持3週；截止後不替換連結指向的模型。

在連結尚未提供／遠端下載未驗證前，`*.pending-url.zip`與`output/submission/r15922a14/`都是準備中的提交件，不能宣稱已完成遠端驗收。`download.sh --final-model-archive ...`的本機測試只證明壓縮檔可用，不代表網路託管已驗證。

報告`report.pdf`已獨立完成，可先閱讀；提交檔案不會由程式自動送到課程平台。
