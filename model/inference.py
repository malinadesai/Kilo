import warnings
import bilby
from bilby.core.prior import Uniform, DeltaFunction
from bilby.core.likelihood import GaussianLikelihood
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

num_points = 121

def cast_as_bilby_result(samples, truth):
    injections = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
    injections['log10_mej'] = float(truth.numpy()[0])
    injections['log10_vej'] = float(truth.numpy()[1])
    injections['log10_Xlan'] = float(truth.numpy()[2])

    posterior = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
    samples_numpy = samples.numpy()
    posterior['log10_mej'] = samples_numpy.T[0].flatten()
    posterior['log10_vej'] = samples_numpy.T[1].flatten()
    posterior['log10_Xlan'] = samples_numpy.T[2].flatten()
    posterior = pd.DataFrame(posterior)
    
    return bilby.result.Result(
        label="test_data",
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=list(injections.keys()),
        priors=priors
    )

def live_plot_samples(samples, truth):
    print(truth)
    clear_output(wait=True)
    sleep(0.5)
    figure = corner.corner(
        samples.numpy(), quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        labels=["Mass of Ejecta", "Vel. of Ejecta", "Lan. Fraction"],
        truth=truth,
    )
    corner.overplot_lines(figure, truth, color="C1")
    corner.overplot_points(figure, truth[None], marker="s", color="C1")

def ppplot(data_loader):
    results = []
    for idx, (shift_test, shift_orig, data_test, data_orig) in enumerate(data_loader):
        data_test = data_test.reshape((-1,)+data_test.shape[2:])
        shift_test = shift_test.reshape((-1,)+shift_test.shape[2:])
        with torch.no_grad():
            samples = flow.sample(1000, context=data_test[0].reshape((1, 3, num_points)))
        results.append(
            cast_as_bilby_result(samples.cpu().reshape(1000,3), shift_test[0][0].cpu()[...,0:3]))
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bilby.result.make_pp_plot(results, save=False, keys=['log10_mej', 'log10_vej', 'log10_Xlan'])

def comparison_plot(num, num_params, varied_dict, injparam_dict, multinest_dict, fixed_num=0, fixed_dict = {}, fixed_available=False, plot_range=None):
    nsamples = len(multinest_dict['multinest_result{}'.format(num)].samples)

    # get the varied result
    with torch.no_grad():
        samples = flow.sample(nsamples, context=varied_dict['var_data{}'.format(num)][0].reshape((1, 3, num_points)))
    samples = samples.cpu().reshape(nsamples,3)
    truth = varied_dict['var_param{}'.format(num)].cpu()[...,0:3]
    injections = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
    injections['log10_mej'] = truth.numpy()[0][0][0]
    injections['log10_vej'] = truth.numpy()[0][0][1]
    injections['log10_Xlan'] = truth.numpy()[0][0][2]
    posterior = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
    samples_numpy = samples.numpy()
    posterior['log10_mej'] = samples_numpy.T[0].flatten()
    posterior['log10_vej'] = samples_numpy.T[1].flatten()
    posterior['log10_Xlan'] = samples_numpy.T[2].flatten()
    posterior = pd.DataFrame(posterior)
    flow_result = bilby.result.Result(
            label="test_data",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=['log10_vej', 'log10_mej', 'log10_Xlan'],
            priors=priors
        )

    # get the fixed result

    if fixed_available == True:
        with torch.no_grad():
            fixed_samples = flow.sample(nsamples, context=fixed_dict['fix_data{}'.format(fixed_num)][0].reshape((1, 3, num_points)))
        fixed_samples = fixed_samples.cpu().reshape(nsamples,3)
        fixed_truth = fixed_dict['fix_param{}'.format(fixed_num)].cpu()[...,0:3]
        fixed_injections = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
        fixed_injections['log10_mej'] = fixed_truth.numpy()[0][0][0]
        fixed_injections['log10_vej'] = fixed_truth.numpy()[0][0][1]
        fixed_injections['log10_Xlan'] = fixed_truth.numpy()[0][0][2]
        fixed_posterior = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
        fixed_samples_numpy = fixed_samples.numpy()
        fixed_posterior['log10_mej'] = fixed_samples_numpy.T[0].flatten()
        fixed_posterior['log10_vej'] = fixed_samples_numpy.T[1].flatten()
        fixed_posterior['log10_Xlan'] = fixed_samples_numpy.T[2].flatten()
        fixed_posterior = pd.DataFrame(fixed_posterior)
        fixed_flow_result = bilby.result.Result(
                label="test_data_fixed",
                injection_parameters=fixed_injections,
                posterior=fixed_posterior,
                search_parameter_keys=['log10_vej', 'log10_mej', 'log10_Xlan'],
                priors=priors
            )
        # plot the figure
        fig = bilby.result.plot_multiple(
        [flow_result, 
         fixed_flow_result, 
         multinest_dict['multinest_result{}'.format(num)],
        ],
        labels=['Shifted Data, \nFlow with Similarity \nRep. ({} params)'.format(num_params), 
                'Fixed Data, \nFlow with Similarity \nRep. ({} params)'.format(num_params), 
                'PyMultiNest Sampling', 
               ],
        truth=injparam_dict['injection_parameters{}'.format(num)],
        corner_labels = ['$\log_{{10}}(M_{{ej}})$', '$\log_{{10}}(V_{{ej}})$', '$\log_{{10}}(X_{{lan}})$'],
        #colours = ['royalblue', 'r', 'mediumseagreen', 'black'],
        colours = ['royalblue', 'r', 'mediumseagreen'],
        quantiles=(0.16, 0.84),
        titles=True,
        save=False, 
        range = plot_range
        )
        plt.show()

    else: 
        # plot the figure
        fig = bilby.result.plot_multiple(
        [flow_result, 
         multinest_dict['multinest_result{}'.format(num)], 
        ],
        labels=['Shifted Data, \nFlow with Similarity \nRep. ({} params)'.format(num_params), 
                'PyMultiNest Sampling', 
               ],
        truth=injparam_dict['injection_parameters{}'.format(num)],
        corner_labels = ['$\log_{{10}}(M_{{ej}})$', '$\log_{{10}}(V_{{ej}})$', '$\log_{{10}}(X_{{lan}})$'],
        colours = ['royalblue', 'mediumseagreen'],
        quantiles=(0.16, 0.84),
        titles=True,
        save=False,
        range = plot_range
        )
        plt.show()
