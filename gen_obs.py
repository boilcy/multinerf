"""Generate observe image script."""

import os

from absl import app
import gin
from internal import configs
from internal import datasets
import jax

configs.define_common_flags()
jax.config.parse_flags_with_absl()

def main(unused_argv):

  config = configs.load_config(save_config=False)
  data_dir = config.data_dir
  colmap_dir = os.path.join(data_dir, 'sparse/0/')

  obs_dir = config.render_dir
  datasets.NeRFSceneManager(colmap_dir).generate_obs(obs_dir)
  
if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)