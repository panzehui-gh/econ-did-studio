# 提出前チェックリスト（2026-08-09 時点）

この表は、配布された Word 文書に記載された要件と、現在のローカル成果物を
照合したものです。「要確認」はコード上の不足ではなく、GitHub または提出画面で
本人の操作・情報が必要な項目です。

| 要件 | 現在の証拠 | 状態 |
|---|---|---|
| 他者にとって有用なソフトウェア | DID/DDD のデータ整理、役割候補、推定、診断、出力を一体化 | 完了 |
| README に目的 | `README.md` / `README_ja.md` 冒頭 | 完了 |
| 明確な例・可視化 | 合成デモ、CLI 例、検証図、任意の推移図・効果/95%CI図 | 完了 |
| インストールと最短利用法 | 両 README の「Fastest start / 最短の起動方法」 | 完了 |
| 開発環境・テスト方法 | 両 README の Development / 開発節 | 完了 |
| CI バッジ | 両 README 冒頭 | 完了（push 後の実行結果は要確認） |
| 単体テスト | `pytest`: 23 passed、パッケージカバレッジ 86% | 完了 |
| GitHub Actions | `.github/workflows/ci.yml`、Python 3.12、pytest + coverage | 構成完了（リモートで緑表示を要確認） |
| 意味のあるコミット履歴 | コア、UI/CLI、テスト/CI、文書を分けた既存コミット＋今回の追加 | 完了 |
| ライセンス | MIT `LICENSE` | 完了 |
| 著作権・秘密情報 | 独自コード、合成データ、機密パターン検索で検出なし | 完了 |
| 2ページ以内の LaTeX PDF | `report/report_ja.tex` → 2ページ PDF、全ページ目視確認済み | 完了 |
| 報告書：目的・工夫 | 日語報告の第1・2節 | 完了 |
| 報告書：客観的検証 | 既知効果40推定＋独立回帰200件＋脏数据60件 | 完了 |
| Matplotlib のベクトル図 | `docs/cross_validation/manual_vs_one_click.pdf` を LaTeX に埋込 | 完了 |
| AI 利用の説明 | 日語報告第4節に OpenAI Codex (GPT-5) と検証方法 | 完了 |
| 氏名・学籍番号 | 日語 PDF は記入欄のテンプレート | **本人が記入** |
| GitHub URL | `https://github.com/panzehui-gh/econ-did-studio` を仮定 | **実在・公開/共有状態を確認** |
| private の場合の協力者 | `sotetsuk` | **GitHub 上で追加** |
| GitHub Actions の成功 | push 後の Actions タブ | **緑のチェックを確認** |
| 提出フォーム | GitHub URL と最終 PDF | **締切前に本人が送信** |

## 最終操作

1. `report/report_ja.tex` の氏名・学籍番号と GitHub URL を確定し、再コンパイルする。
2. 変更を GitHub に push する。
3. private リポジトリなら `sotetsuk` を collaborator に追加する。
4. Actions が成功したことと README のバッジが緑であることを確認する。
5. GitHub URL と2ページの PDF を提出フォームへ送信する。
