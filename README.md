# Datamining-Group1

動作環境
・MacBook Pro (13-inch, 2016, Two Thunderbolt 3 ports)

環境構築
以下の が必要
・python3 version3.7.5
・dtreeviz version0.8.1
・sklearn version0.21.3

1. python3
pythonは各自環境設定行なってください

2.dtreeviz
ターミナルで以下のコードを実行してください

% pip install dtreeviz

得られる決定木の可視化に必要なモジュールです

3.  sklearn
ターミナルで以下のコードを入力してください
% pip install sklearn

データセット
githubにソースコードとドキュメントをアップロードしております。
今回教師データはプログラム上で用意するようにしています。

実行方法
DecisionTree.pyもしくはDecisionTree_All.pyをダウンロードし、pythonで実行してください。

DecisionTree.pyは、備え付けのデータセットを決定木分析し、結果を可視化するコードです。
DecisionTree_All.pyは、機能としてはDecisionTree.pyとほとんど一緒ですが、学習データを任意で選択できるようにしています。

判断方法
実行結果が1に近いほど分類精度が高いです。
