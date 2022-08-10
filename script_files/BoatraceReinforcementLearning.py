#ボートレースの強化学習を行えるようなエージェント，環境生成のための関数・クラスをまとめたスクリプトファイル

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np



#Q関数
class QFunction(chainer.Chain):#chainer.Chain:link Lやfunction Fをまとめて管理するものらしい，親クラスとして提供されているためニューラルネットを定義する際はこれを継承させるらしい．

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        #n_actions:行動の種類数
        #obs_size :入力データの次元数
        #l1,l2等々は層を表してる，l2が出力層
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)#Lはchainer.linksです（インポート時にas　L　でインポートしてる）
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)#Lはchainer.linksです（インポート時にas　L　でインポートしてる）
            self.l2 = L.Linear(n_hidden_channels, n_actions)#Lはchainer.linksです（インポート時にas　L　でインポートしてる）

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
            [翻訳]
            x（ndarrayまたはchainer.Variable）：観測値
            test（bool）：テストモードかどうかを示すフラグ
        """
        h = F.tanh(self.l0(x))#F は　chainer.functions
        h = F.tanh(self.l1(h))#F は　chainer.functions
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))
