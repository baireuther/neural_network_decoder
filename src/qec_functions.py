# ========================================================================
# This file is part of a decoder for small stabilizer codes, in particular
# color and surface codes, based on a combination of recurrent and
# feedforward neural networks.
#
# Copyright 2017-2018 Paul Baireuther. All Rights Reserved.
# ========================================================================


# # # LIBRARIES # # #

# Third party libraries
import scipy.optimize as optim
import matplotlib.pyplot as plt
import numpy as np


# # # FUNCTIONS # # #

def decay_single_param(t, p_logical):
  """ Function that models exponential (fidelity) decay controlled
      by a single parameter, the decay rate.

  Input
  -----
  t -- Time, a positive float or integer.
  p_logical -- Decay rate, a positive float number.

  Output
  ------
  fidelity -- The logical fidelity, a float number between 0 and 1.
  """
  fidelity = (1 + (1 - 2 * p_logical)**t) / 2.
  return fidelity


def decay_two_param(t, p_logical, t0):
  """ Function that models exponential (fidelity) decay controlled
      by a two parameters, the decay rate and a time-offset.

  Input
  -----
  t -- Time, a positive float or integer.
  p_logical -- Decay rate, a positive float number.
  t0 -- Time offset, a float number.

  Output
  ------
  fidelity -- The logical fidelity, a float number between 0 and 1.
  """
  return (1 + (1 - 2 * p_logical)**(t - t0)) / 2.


def resample_array(data):
  """ Function to resample an array with redraw.

  Input
  -----
  data -- A numpy array.

  Output
  ------
  data_resampled -- A numpy array.
  """
  N = len(data[0])
  idcs = N * np.random.rand(N)
  idcs = idcs.astype(int)
  data = np.array(data)
  data_resampled = data[:, idcs]
  return data_resampled


def calc_plog(data, p0=(0.001,), bounds=((0.00001), (.2))):
  """ Function to calculate the logical error rate.

  Input
  -----
  data -- List of lists.
  p0 -- A starting point for the fits.
  bounds -- Boundaries for the fitting, tuple of tuples of floats.

  Output
  ------
  plog -- The logical error rate, a float number.
  """

  # It is possible that the batch does not contain fidelities
  # for all steps, hence we need a list with all steps for which
  # predictions exist (we call it 'steps').
  steps, data_nonzero = [], []
  fids = []

  # In the following we assume that the first step is s = 1.
  for s in range(1, len(data) + 1):
    dat = data[s - 1]
    if len(dat) != 0:
      # Non-trivial data points
      steps.append(s)
      data_nonzero.append(dat)
      # Fidelities
      fids.append(np.mean(dat))

  # We fit a decay curve to the non-tivial data.
  popt, pcov = optim.curve_fit(decay_single_param, steps, fids,
                               p0, bounds=bounds)
  plog = popt[0]
  return plog


def calc_stats(data, bootstrap, p0, n_sampling=100,
               verbose=False, visualize_bootstrapping=False,
               lower_bounds=(0, 0), upper_bounds=(20, 10)):
  """ Calculates the logical error rate and error bars via bootstrapping.

  Input
  -----
  data -- A list of lists.
  bootstrap -- If True, error bars will be calculated using bootstrapping
  p0 -- Start values for fitting logical error rate and time offset, a tuple
        of float numbers.
  n_sampling -- The number of redraws during bootstrapping.
  verbose -- If True, more output will be printed.
  visualize_bootstrapping -- If True, plots with the distributions of the
                             bootstrapping will be shown.
  lower_bounds -- Lower bounds of the fitting paramters (tuple of floats).
  upper_bounds -- Upper bounds of the fitting paramters (tuple of floats).

  Output
  ------
  res_dict -- A dictionary with the results.

 """

  # It is possible that the batch does not contain fidelities for all steps,
  # hence we need a list with all steps for which predictions excist (we call
  # it 'steps').
  steps, data_nonzero = [], []
  fids, rs_means_l, plogs_bs = [], [], []

  # In the following we assume that the first step is s = 1.
  for s in range(1, len(data) + 1):
    dat = data[s - 1]
    if len(dat) != 0:
      # non-trivial data points
      steps.append(s)
      data_nonzero.append(dat)
      # fidelities
      fids.append(np.mean(dat))

  # Fit decay curve to the non-tivial data.
  popt, pcov = optim.curve_fit(
      decay_two_param, steps, fids,
      p0, bounds=(lower_bounds, upper_bounds))
  plog, x0 = popt[0], popt[1]
  if x0 > 0.99 * upper_bounds[1]:
    print("WARNING: x0 is larger than", upper_bounds[1],
          "the fitting algorithm fails.")
  if plog > 0.99 * upper_bounds[0]:
    print("WARNING: plog is larger than", upper_bounds[0], "\%, the " +
          "fitting algorithm fails.")

  # Using bootstrapping to calculate error bars.
  if bootstrap:
    # Bootstrapping for error bars of fidelities and logical error rate p.
    # It is assumed that all steps -- if they exist -- have the same
    # number of entries.
    ls = [len(dat) for dat in data_nonzero]
    if min(ls) != max(ls):
      raise ValueError("The array with the fidelities is not uniform")

    for n in range(n_sampling):
      rs_data = resample_array(data_nonzero)
      rs_means = np.array(np.mean(rs_data, axis=1))
      rs_means_l.append(rs_means)

      popt, pcov = optim.curve_fit(decay_two_param, steps, rs_means,
                                   p0, bounds=(lower_bounds, upper_bounds))
      plogs_bs.append(popt[0])
    if visualize_bootstrapping:
      for el1, el2 in zip(np.array(rs_means_l).transpose(), fids):
        plt.hist(el1, bins=15)
        plt.vlines(el2, 0, n_sampling / 5, linestyles="dashed")
        plt.show()
      print("logical error rate bootstrap")
      plt.hist(plogs_bs, bins=15)
      plt.vlines(plog, 0, n_sampling / 5, linestyles="dashed")
      plt.show()
    fids_sdv_bs = np.std(rs_means_l, axis=0)
    plog_sdv_bs = np.std(plogs_bs)

  if bootstrap:
    res_dict = {'steps': steps,
                'fids': fids, 'fids_sdv_bs': fids_sdv_bs,
                'plog': plog, 'plog_sdv_bs': plog_sdv_bs, 'x0': x0}
    if verbose:
      print("logical error rate:", round(plog * 100, 5),
            "+-", round(plog_sdv_bs * 100, 5), "%")
      print("t0 offset", round(x0, 3))
  else:
    res_dict = {'steps': steps, 'fids': fids, 'plog': plog, 'x0': x0}
    if verbose:
      print("logical error rate:", round(plog * 100, 5))
      print("t0 offset", round(x0, 3))

  return res_dict


def plot_fids(stats_dict, error_bars=False, fmin=.5, title="",
              fname=None):
  """ Function to visualize the fidelity decay curves.

  Input
  -----
  stats_dict -- A dictionary with the results, as returned from calc_stats.
  error_bars -- If True, error bars are displayed.
  fmin -- Lower bound of the y-axis, float in the interval (0, 1).
  title -- A string as caption of the plot.
  fname -- If set, the plot will be saved to this directory + filename.
  """

  fig, ax = plt.subplots(1, 1, figsize=(7, 5))

  steps, fids = stats_dict['steps'], stats_dict['fids']
  plog, x0 = stats_dict['plog'], stats_dict['x0']

  if error_bars:
    fids_sdv_bs = stats_dict['fids_sdv_bs']
    plog_sdv_bs = stats_dict['plog_sdv_bs']

  xs = steps
  if error_bars:
    ax.errorbar(x=xs, y=fids,
                yerr=np.array(fids_sdv_bs),
                fmt='.', label='neural network', color='#7599f4', ms=7)
  else:
    ax.plot(xs, fids, '.', label='neural network', color='#7599f4', ms=7)

  xs = np.arange(1, max(steps))
  if error_bars:
    ax.plot(xs, [decay_two_param(x, plog, x0) for x in xs], '--',
            color='#7599f4', lw=2, label='logical p = ' +
            str(round(plog * 100, 6)) + "+-" +
            str(round(plog_sdv_bs * 100, 6)) + "%.")
  else:
    ax.plot(xs, [decay_two_param(x, plog, x0) for x in xs], '--',
            color='#7599f4', lw=2, label='logical p = ' +
            str(round(plog * 100, 6)) + "%.")

  for tl in ax.get_xticklabels():
    tl.set_color('k')
    tl.set_size(16)
  for tl in ax.get_yticklabels():
    tl.set_color('k')
    tl.set_size(16)
  ax.set_xlim(0, max(steps))
  ax.set_ylim(fmin, 1)
  ax.set_xlabel('cycles', color='k', fontsize=20)
  ax.legend(loc='lower left')
  ax.locator_params(axis='x', nbins=5)
  ax.locator_params(axis='y', nbins=5)

  ax.set_ylabel('fidelity', color='k', fontsize=20)
  ax.set_title(title, fontsize=16)
  fig.tight_layout()
  plt.show()

  if fname:
    fig.savefig(fname, bbox_inches='tight')
