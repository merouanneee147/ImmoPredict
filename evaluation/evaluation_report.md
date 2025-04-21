
# Rapport d'�valuation du mod�le de pr�diction immobili�re

## 1. Informations g�n�rales

- **Type de mod�le**: RandomForestRegressor
- **Nombre d'observations**: 116089
- **Nombre de features**: 19
- **Date d'�valuation**: 2025-04-21 02:47:06

## 2. M�triques de performance

- **RMSE (Root Mean Squared Error)**: 13585.23
- **MAE (Mean Absolute Error)**: 563.16
- **R� (Coefficient de d�termination)**: 0.9987

## 3. Validation crois�e (5-fold)

- **RMSE moyen**: 13121.58
- **�cart-type RMSE**: 2689.99
- **Scores par fold**: 9965.12, 15457.70, 15907.53, 9786.51, 14491.02

## 4. Analyse des r�sidus

- **Moyenne des r�sidus**: 74.14
- **�cart-type des r�sidus**: 13585.03
- **R�sidu minimum**: -583916.70
- **R�sidu maximum**: 2684014.60

## 5. Analyse par segment de prix

| Segment   |   ('AbsoluteError', 'mean') |   ('AbsoluteError', 'std') |   ('AbsoluteError', 'min') |   ('AbsoluteError', 'max') |   ('PercentageError', 'mean') |   ('PercentageError', 'std') |   ('PercentageError', 'min') |   ('PercentageError', 'max') |   ('Actual', 'count') |   ('Actual', 'mean') |
|:----------|----------------------------:|---------------------------:|---------------------------:|---------------------------:|------------------------------:|-----------------------------:|-----------------------------:|-----------------------------:|----------------------:|---------------------:|
| 0-25%     |                     117.872 |                    298.389 |                          0 |            12229.7         |                     0.195614  |                     0.643783 |                            0 |                      37.6037 |                 29018 |              79454.1 |
| 25-50%    |                     127.999 |                    406.06  |                          0 |            34821.1         |                     0.0768216 |                     0.256704 |                            0 |                      23.2141 |                 29026 |             166762   |
| 50-75%    |                     195.791 |                    377.419 |                          0 |            13969.5         |                     0.0749758 |                     0.139457 |                            0 |                       5.2715 |                 29009 |             260269   |
| 75-100%   |                    1810.23  |                  27095.5   |                          0 |                2.68401e+06 |                     0.123816  |                     0.469832 |                            0 |                      23.5225 |                 29036 |             656352   |

## 6. Conclusion

Le mod�le RandomForestRegressor atteint un R� de 0.9987, ce qui signifie qu'il explique environ 99.9% de la variance dans les prix immobiliers. Le RMSE de 13585.23 indique l'erreur moyenne en unit�s de la variable cible.

L'analyse des r�sidus montre que le mod�le a tendance � sous-estimer les prix.

L'analyse par segment r�v�le que le mod�le est 0-25% pour les propri�t�s dans le segment de prix 0-25%.

## 7. Recommandations

1. Le mod�le performe bien, mais pourrait b�n�ficier d'une mise � jour r�guli�re avec de nouvelles donn�es.
2. La distribution des erreurs est relativement stable � travers les diff�rentes gammes de prix.
3. Les performances sont relativement homog�nes � travers les diff�rents segments de prix.
