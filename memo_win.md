# アプリの引っ越し
## ライセンス
* Sount it
* Adobe Photoshop
* TMPG Enc&Wrok
* デジカメで同時プリント
  * フィルムのコピー
    * フィルムを旧環境でバックアップして新環境でリストアしてリストア先をものとフォルダとすり替える。フィルムの場所は共有ドキュメント\Aisoft\HomeDPE\Films
  * フィルムケースのコピー
    * ProgramData\AISOFT\HomeDPE\AiFilm.db0 をコピーして、パス名をバイナリ編集( vi -b )する。

## コピー
* XP のネットワークドライブが見えないときはXPのウイルスソフトをOFFにする
* Exmax
  * フォルダと.DAAと.DATを上書き
  * 設定→メーラ設定→表示で HTMLメールを表示、警告ポップアップは"操作"で添付ファイルのセキュリティのチェックを外す
  * edmax.ini のWarningHtmlSecurity=1 -> 0
  * でも、HTMLはセキュリティ上アクセスしないほうがよい。メニューバーのアイコンを押せばHTML表示に変わる。
* 筆自慢
  * accugnt5.dll のwindoes\SystemWoW64へのコピー
  * C:\windows\Mets\*.ocxの６個のファイルを登録。コマンドラインから Regsrv32 C:\windows\Mets\*.ocx。

## 再インストール
* WinMerge
* ffftp
* CD/DVDライティングソフト
* (Open)Office
* Degital Photo Pro
* Easy-Phot Print
* GV
* VLC media player
* lhaplus
* Play Memory Home(SONY)


# 購入音楽コンテンツ

# 環境設定

## Printerの設定



## キーボードの設定
* 106キーボードを英語にする。
http://qiita.com/shimizu14/items/000cceb9e72a492b9176
* かな漢対応
  * 変換キーの設定
  * かなモードで半角スペース

## xyzzy
* 設定変更
  * C:\Program Files (x86)\xyzzy\site-lisp\siteinit.el に Lispファイルを書く
  * 色の設定とかは C:\Program Files (x86)\xyzzy\usr\papa\wxp の下にファイルが作られる
  * Windows7 はProgram Files(x86) の下にxyzzyフォルダを置くと、システムがファイルを作成できず、環境が保存できない
  * ファイル名の関連付けで、xyzzyが既定のプログラムとして追加できない場合は、レジストリを変更するらしい
    http://cheese999.blog.so-net.ne.jp/2013-02-11

## Atom 環境整備
* grep は標準メニューのプロジェクト内検索
* tagジャンプはatom標準のSymbols-viewsよりかはatom-gtags
 * gnu-globalはcygwinでビルドしなおしてインストール、cygwinパッケージとして djgpp,  libncurse-devel をインストールしておく。
 * ctags はcygwinから標準でインストールできる

* atom-gtags の課題
  * うちのマシンではデフォルトのalt-1 ～4等が動作しない。
  * 再起動すると少し安定するようなきがする。
  * やっぱ navi-backward でもどれないことがある。
  * tag位置がずれやすい。
  * 編集後に自動でgtags が動くようにも見えるが、合わないのでだめなのか？

* Symboles-vies は大丈夫そう。

* Symboles-tree-view を入れてみた。ctrl-alt-o でトグル

* 改行幅は環境設置->エディター設定->行の高さで1.3くらいを設定
* 設定のところで、インデントガイドと空白文字の表示を設定

* 追加パッケージ
  * color-picker
  *　pigments
* 追加テーマ
  * rain
    * 場所はC:\Users\papa\.atom\packages
    * rain-syntax/styles/syntax-variables.less を編集すれば、色が変えられるが、書式が変更に
      なっているので、編集済みの古いものをパッケージフォルダごと差し替える。残骸が残っていると
      おかしなことになる。

* 便利なショートカット
  * Ctrl-Shift-p ：コマンドメニューと探索窓
  * Ctrl-T：ファイルの検索は
  * Ctrl-,：設定メニュー
  * Ctrl-\ : ディレクトリーツリーを隠す

## Ricty Diminished フォント

## Chrome
* Chrome のフォントで Fixed をRictyに、ゴシックをメイリオに変えた。
    場所はここchrome://settings/fonts

# その他
