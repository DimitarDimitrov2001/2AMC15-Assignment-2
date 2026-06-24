# Final Evaluation Results

Cells report `mean +/- std` across retained training seeds; `var` shows the seed variance used to compute the standard deviation.
Average total training collisions is computed from per-episode training histories because final greedy collision counts are not emitted in `evaluation_summary.txt`.

<table>
  <thead>
    <tr>
      <th rowspan="3">Experiment Setting</th>
      <th colspan="8">Simple Grid</th>
      <th colspan="8">Medium Difficulty Grid</th>
      <th colspan="8">Difficult Grid</th>
    </tr>
    <tr>
      <th colspan="2">Eval return</th>
      <th colspan="2">Success rate</th>
      <th colspan="2">Train collisions</th>
      <th colspan="2">Eval length</th>
      <th colspan="2">Eval return</th>
      <th colspan="2">Success rate</th>
      <th colspan="2">Train collisions</th>
      <th colspan="2">Eval length</th>
      <th colspan="2">Eval return</th>
      <th colspan="2">Success rate</th>
      <th colspan="2">Train collisions</th>
      <th colspan="2">Eval length</th>
    </tr>
    <tr>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
      <th>DQN</th>
      <th>Dueling DQN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Baseline</strong></td>
      <td>0.980 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.980 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>21024.0 +/- 3064.5<br><small>var=9390967.5</small></td>
      <td>19166.6 +/- 1748.5<br><small>var=3057301.3</small></td>
      <td>21.0 +/- 0.0<br><small>var=0.0</small></td>
      <td>21.2 +/- 0.4<br><small>var=0.2</small></td>
      <td>0.935 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.935 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>18448.6 +/- 1324.9<br><small>var=1755484.8</small></td>
      <td>17252.0 +/- 1165.7<br><small>var=1358851.0</small></td>
      <td>66.2 +/- 0.4<br><small>var=0.2</small></td>
      <td>66.0 +/- 0.0<br><small>var=0.0</small></td>
      <td>0.625 +/- 0.629<br><small>var=0.395</small></td>
      <td>0.907 +/- 0.001<br><small>var=0.000</small></td>
      <td>0.800 +/- 0.447<br><small>var=0.200</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>67246.6 +/- 15283.9<br><small>var=233598848.3</small></td>
      <td>57549.4 +/- 8008.3<br><small>var=64132471.3</small></td>
      <td>176.0 +/- 181.1<br><small>var=32809.5</small></td>
      <td>93.8 +/- 0.8<br><small>var=0.7</small></td>
    </tr>
    <tr>
      <td><strong>No LiDAR</strong></td>
      <td>0.977 +/- 0.001<br><small>var=0.000</small></td>
      <td>0.977 +/- 0.001<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>35556.2 +/- 9589.4<br><small>var=91957073.2</small></td>
      <td>38072.0 +/- 10380.4<br><small>var=107753468.0</small></td>
      <td>23.6 +/- 0.9<br><small>var=0.8</small></td>
      <td>23.6 +/- 0.9<br><small>var=0.8</small></td>
      <td>-0.500 +/- 0.000<br><small>var=0.000</small></td>
      <td>-0.500 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>77488.2 +/- 5829.0<br><small>var=33977667.2</small></td>
      <td>80581.0 +/- 4815.6<br><small>var=23189658.0</small></td>
      <td>500.0 +/- 0.0<br><small>var=0.0</small></td>
      <td>500.0 +/- 0.0<br><small>var=0.0</small></td>
      <td>-0.500 +/- 0.000<br><small>var=0.000</small></td>
      <td>-0.500 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>145864.2 +/- 28134.4<br><small>var=791546978.2</small></td>
      <td>118309.2 +/- 4552.3<br><small>var=20723703.2</small></td>
      <td>500.0 +/- 0.0<br><small>var=0.0</small></td>
      <td>500.0 +/- 0.0<br><small>var=0.0</small></td>
    </tr>
    <tr>
      <td><strong>Stochastic ($\sigma=0.5$)</strong></td>
      <td>0.972 +/- 0.004<br><small>var=0.000</small></td>
      <td>0.971 +/- 0.003<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>16624.4 +/- 2116.3<br><small>var=4478640.3</small></td>
      <td>15083.0 +/- 1510.9<br><small>var=2282891.5</small></td>
      <td>26.7 +/- 2.0<br><small>var=4.1</small></td>
      <td>27.2 +/- 2.9<br><small>var=8.5</small></td>
      <td>0.927 +/- 0.002<br><small>var=0.000</small></td>
      <td>0.926 +/- 0.002<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>15033.0 +/- 2279.2<br><small>var=5194640.0</small></td>
      <td>13778.6 +/- 1057.0<br><small>var=1117177.3</small></td>
      <td>72.9 +/- 1.7<br><small>var=2.8</small></td>
      <td>73.9 +/- 1.8<br><small>var=3.3</small></td>
      <td>0.887 +/- 0.007<br><small>var=0.000</small></td>
      <td>0.849 +/- 0.058<br><small>var=0.003</small></td>
      <td>1.000 +/- 0.000<br><small>var=0.000</small></td>
      <td>0.980 +/- 0.045<br><small>var=0.002</small></td>
      <td>44816.8 +/- 5553.3<br><small>var=30839237.7</small></td>
      <td>42024.6 +/- 3279.6<br><small>var=10755822.3</small></td>
      <td>109.0 +/- 7.1<br><small>var=51.0</small></td>
      <td>126.6 +/- 20.2<br><small>var=406.6</small></td>
    </tr>
  </tbody>
</table>
