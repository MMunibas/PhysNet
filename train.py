#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import logging
import string
import random
from shutil import copyfile
from datetime import datetime
from neural_network.NeuralNetwork import *
from neural_network.activation_fn import *
from training.Trainer        import *
from training.DataContainer import *
from training.DataProvider  import *
from training.DataQueue     import *

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

logging.basicConfig(filename='train.log',level=logging.DEBUG)

#define command line arguments
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
parser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
parser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
parser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
parser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int,   help="use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int,   help="use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")
parser.add_argument("--dataset", type=str,   help="file path to dataset")
parser.add_argument("--num_train", type=int,   help="number of training samples")
parser.add_argument("--num_valid", type=int,   help="number of validation samples")
parser.add_argument("--seed", default=42, type=int,   help="seed for splitting dataset into training/validation/test")
parser.add_argument("--max_steps", type=int,   help="maximum number of training steps")
parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate used by the optimizer")
parser.add_argument("--max_norm", default=1000.0, type=float, help="max norm for gradient clipping")
parser.add_argument("--ema_decay", default=0.999, type=float, help="exponential moving average decay used by the trainer")
parser.add_argument("--keep_prob", default=1.0, type=float, help="keep probability for dropout regularization of rbf layer")
parser.add_argument("--l2lambda", type=float, help="lambda multiplier for l2 loss (regularization)")
parser.add_argument("--nhlambda", type=float, help="lambda multiplier for non-hierarchicality loss (regularization)")
parser.add_argument("--decay_steps", type=int, help="decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate", type=float, help="factor with which the learning rate gets multiplied by every decay_steps steps")
parser.add_argument("--batch_size", type=int, help="batch size used per training step")
parser.add_argument("--valid_batch_size", type=int, help="batch size used for going through validation_set")
parser.add_argument('--force_weight',  default=52.91772105638412, type=float, help="this defines the force contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--charge_weight', default=14.399645351950548, type=float, help="this defines the charge contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--dipole_weight', default=27.211386024367243, type=float, help="this defines the dipole contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
parser.add_argument('--summary_interval', type=int, help="write a summary every N steps")
parser.add_argument('--validation_interval', type=int, help="check performance on validation set every N steps")
parser.add_argument('--save_interval', type=int, help="save progress every N steps")
parser.add_argument('--record_run_metadata', type=int, help="records metadata like memory consumption etc.")

#if no command line arguments are present, config file is parsed
config_file='config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@"+config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

#create directories
#a unique directory name is created for this run based on the input
if args.restart is None:
    directory=datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + id_generator() +"_F"+str(args.num_features)+"K"+str(args.num_basis)+"b"+str(args.num_blocks)+"a"+str(args.num_residual_atomic)+"i"+str(args.num_residual_interaction)+"o"+str(args.num_residual_output)+"cut"+str(args.cutoff)+"e"+str(args.use_electrostatic)+"d"+str(args.use_dispersion)+"l2"+str(args.l2lambda)+"nh"+str(args.nhlambda)+"keep"+str(args.keep_prob)
else:
    directory=args.restart

logging.info("creating directories...")
if not os.path.exists(directory):
    os.makedirs(directory)
best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)
log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
best_checkpoint = os.path.join(best_dir, 'best_model.ckpt')
step_checkpoint = os.path.join(log_dir,  'model.ckpt')

#write config file (to restore command line arguments)
logging.info("writing args to file...")
with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--'+ arg + '='+ str(getattr(args, arg)) + "\n")

#load dataset
logging.info("loading dataset...")
data = DataContainer(args.dataset)

#generate DataProvider (splits dataset into training, validation and test set based on seed)
data_provider = DataProvider(data, args.num_train, args.num_valid, args.batch_size, args.valid_batch_size, seed=args.seed)

#create neural network
logging.info("creating neural network...")
nn = NeuralNetwork(F=args.num_features,           
                   K=args.num_basis,                
                   sr_cut=args.cutoff,              
                   num_blocks=args.num_blocks, 
                   num_residual_atomic=args.num_residual_atomic,
                   num_residual_interaction=args.num_residual_interaction,
                   num_residual_output=args.num_residual_output,
                   use_electrostatic=(args.use_electrostatic==1),
                   use_dispersion=(args.use_dispersion==1),
                   s6=args.grimme_s6,
                   s8=args.grimme_s8,
                   a1=args.grimme_a1,
                   a2=args.grimme_a2,
                   Eshift=data_provider.EperA_mean,  
                   Escale=data_provider.EperA_stdev,   
                   activation_fn=shifted_softplus, 
                   seed=None,
                   scope="neural_network")

logging.info("prepare training...")
#generate data queues for efficient training
train_queue = DataQueue(data_provider.next_batch, capacity=1000, scope="train_data_queue")
valid_queue = DataQueue(data_provider.next_valid_batch, capacity=args.num_valid//args.valid_batch_size, scope="valid_data_queue")

#get values for training and validation set
Eref_t, Earef_t, Fref_t, Z_t, Dref_t, Qref_t, Qaref_t, R_t, idx_i_t, idx_j_t, batch_seg_t = train_queue.dequeue_op
Eref_v, Earef_v, Fref_v, Z_v, Dref_v, Qref_v, Qaref_v, R_v, idx_i_v, idx_j_v, batch_seg_v = valid_queue.dequeue_op

#calculate all necessary quantities (unscaled partial charges, energies, forces)
Ea_t, Qa_t, Dij_t, nhloss_t = nn.atomic_properties(Z_t, R_t, idx_i_t, idx_j_t)
Ea_v, Qa_v, Dij_v, nhloss_v = nn.atomic_properties(Z_v, R_v, idx_i_v, idx_j_v)
energy_t, forces_t = nn.energy_and_forces_from_atomic_properties(Ea_t, Qa_t, Dij_t, Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
energy_v, forces_v = nn.energy_and_forces_from_atomic_properties(Ea_v, Qa_v, Dij_v, Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)
#total charge
Qtot_t = tf.segment_sum(Qa_t, batch_seg_t)
Qtot_v = tf.segment_sum(Qa_v, batch_seg_v)
#dipole moment vector
QR_t = tf.stack([Qa_t*R_t[:,0], Qa_t*R_t[:,1], Qa_t*R_t[:,2]],1)
QR_v = tf.stack([Qa_v*R_v[:,0], Qa_v*R_v[:,1], Qa_v*R_v[:,2]],1)
D_t = tf.segment_sum(QR_t, batch_seg_t)
D_v = tf.segment_sum(QR_v, batch_seg_v)

#function to calculate loss, mean squared error, mean absolute error between two values
def calculate_errors(val1, val2, weights=1):
    with tf.name_scope("calculate_errors"):
        delta  = tf.abs(val1-val2)
        delta2 = delta**2
        mse    = tf.reduce_mean(delta2)
        mae    = tf.reduce_mean(delta)
        loss   = mae #mean absolute error loss
    return loss, mse, mae

with tf.name_scope("loss"):
    #calculate energy, force, charge and dipole errors/loss
    #energy
    if data.E is not None:
        eloss_t, emse_t, emae_t = calculate_errors(Eref_t, energy_t)
        eloss_v, emse_v, emae_v = calculate_errors(Eref_v, energy_v)
    else:
        eloss_t, emse_t, emae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        eloss_v, emse_v, emae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
    #atomic energies
    if data.Ea is not None:
        ealoss_t, eamse_t, eamae_t = calculate_errors(Earef_t, Ea_t)
        ealoss_v, eamse_v, eamae_v = calculate_errors(Earef_v, Ea_v)
    else:
        ealoss_t, eamse_t, eamae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        ealoss_v, eamse_v, eamae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
    #forces
    if data.F is not None:
        floss_t, fmse_t, fmae_t = calculate_errors(Fref_t, forces_t)
        floss_v, fmse_v, fmae_v = calculate_errors(Fref_v, forces_v)
    else:
        floss_t, fmse_t, fmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        floss_v, fmse_v, fmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
    #charge
    if data.Q is not None:     
        qloss_t, qmse_t, qmae_t = calculate_errors(Qref_t, Qtot_t)
        qloss_v, qmse_v, qmae_v = calculate_errors(Qref_v, Qtot_v)
    else:
        qloss_t, qmse_t, qmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        qloss_v, qmse_v, qmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
    #atomic charges
    if data.Qa is not None:
        qaloss_t, qamse_t, qamae_t = calculate_errors(Qaref_t, Qa_t)
        qaloss_v, qamse_v, qamae_v = calculate_errors(Qaref_v, Qa_v)
    else:
        qaloss_t, qamse_t, qamae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        qaloss_v, qamse_v, qamae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
    #dipole
    if data.D is not None:
        dloss_t, dmse_t, dmae_t = calculate_errors(Dref_t, D_t)
        dloss_v, dmse_v, dmae_v = calculate_errors(Dref_v, D_v)
    else:
        dloss_t, dmse_t, dmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        dloss_v, dmse_v, dmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)

    #define additional variables (such that certain losses can be overwritten)
    eloss_train = eloss_t
    floss_train = floss_t
    qloss_train = qloss_t
    dloss_train = dloss_t
    eloss_valid = eloss_v
    floss_valid = floss_v
    qloss_valid = qloss_v
    dloss_valid = dloss_v

    #atomic energies are present, so they replace the normal energy loss
    if data.Ea is not None:
        eloss_train = ealoss_t
        eloss_valid = ealoss_v

    #atomic charges are present, so they replace the normal charge loss and nullify dipole loss
    if data.Qa is not None:
        qloss_train = qaloss_t
        qloss_valid = qaloss_v
        dloss_train = tf.constant(0.0)
        dloss_valid = tf.constant(0.0)

    #define loss function (used to train the model)
    l2loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 
    loss_t = eloss_train + args.force_weight*floss_train + args.charge_weight*qloss_train + args.dipole_weight*dloss_train + args.nhlambda*nhloss_t + args.l2lambda*l2loss
    loss_v = eloss_valid + args.force_weight*floss_valid + args.charge_weight*qloss_valid + args.dipole_weight*dloss_valid + args.nhlambda*nhloss_v + args.l2lambda*l2loss

#create trainer
trainer  = Trainer(args.learning_rate, args.decay_steps, args.decay_rate, scope="trainer")
with tf.name_scope("trainer_ops"):
    train_op = trainer.build_train_op(loss_t, args.ema_decay, args.max_norm)
    save_variable_backups_op = trainer.save_variable_backups()
    load_averaged_variables_op = trainer.load_averaged_variables()
    restore_variable_backups_op = trainer.restore_variable_backups()

#creates a summary from key-value pairs given a dictionary
def create_summary(dictionary):
    summary = tf.Summary()
    for key, value in dictionary.items():
        summary.value.add(tag=key, simple_value=value)
    return summary

#create summary writer
nn_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())

#create saver
with tf.name_scope("saver"):
    saver = tf.train.Saver(max_to_keep=50)

#save/load best recorded loss (only the best model is saved)
if os.path.isfile(best_loss_file):
    loss_file   = np.load(best_loss_file)
    best_loss   = loss_file["loss"].item()
    best_emae   = loss_file["emae"].item()
    best_ermse  = loss_file["ermse"].item()
    best_fmae   = loss_file["fmae"].item()
    best_frmse  = loss_file["frmse"].item()
    best_qmae   = loss_file["qmae"].item()
    best_qrmse  = loss_file["qrmse"].item()
    best_dmae   = loss_file["dmae"].item()
    best_drmse  = loss_file["drmse"].item()
    best_step   = loss_file["step"].item()
else:
    best_loss  = np.Inf #initialize best loss to infinity
    best_emae  = np.Inf
    best_ermse = np.Inf
    best_fmae  = np.Inf
    best_frmse = np.Inf
    best_qmae  = np.Inf
    best_qrmse = np.Inf
    best_dmae  = np.Inf
    best_drmse = np.Inf
    best_step  = 0.
    np.savez(best_loss_file, loss=best_loss, emae=best_emae,   ermse=best_ermse, 
                                             fmae=best_fmae,   frmse=best_frmse, 
                                             qmae=best_qmae,   qrmse=best_qrmse, 
                                             dmae=best_dmae,   drmse=best_drmse, 
                                             step=best_step)

#for calculating average performance on the training set
def reset_averages():
    return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def update_averages(num, tmploss_avg, tmploss, emse_avg, emse, emae_avg, emae, fmse_avg, fmse, fmae_avg, fmae, 
                    qmse_avg, qmse, qmae_avg, qmae, dmse_avg, dmse, dmae_avg, dmae):
    num += 1
    tmploss_avg += (tmploss-tmploss_avg)/num
    emse_avg += (emse-emse_avg)/num
    emae_avg += (emae-emae_avg)/num
    fmse_avg += (fmse-fmse_avg)/num
    fmae_avg += (fmae-fmae_avg)/num
    qmse_avg += (qmse-qmse_avg)/num
    qmae_avg += (qmae-qmae_avg)/num
    dmse_avg += (dmse-dmse_avg)/num
    dmae_avg += (dmae-dmae_avg)/num
    return num, tmploss_avg, emse_avg, emae_avg, fmse_avg, fmae_avg, qmse_avg, qmae_avg, dmse_avg, dmae_avg

#initialize training set error averages
num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()

#create tensorflow session
with tf.Session() as sess:
    if (args.record_run_metadata > 0):
        run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options  = None
        run_metadata = None

    #start data queues
    coord = tf.train.Coordinator()
    train_queue.create_thread(sess, coord)
    valid_queue.create_thread(sess, coord)

    #initialize variables
    tf.global_variables_initializer().run()

    #restore latest checkpoint
    checkpoint = tf.train.latest_checkpoint(log_dir)
    if checkpoint is not None:
        step = int(checkpoint.split('-')[-1]) #reads step from checkpoint filename
        saver.restore(sess, checkpoint)
        sess.run(trainer.global_step.assign(step))
    else:
        step = 0

    #training loop
    logging.info("starting training...")
    while not coord.should_stop():
        #finish training when maximum number of iterations is reached
        if step > args.max_steps:
            coord.request_stop()
            break

        #perform training step 
        step += 1
        _, tmploss, emse, emae, fmse, fmae, qmse, qmae, dmse, dmae = sess.run([train_op, loss_t, emse_t, emae_t, fmse_t, fmae_t, qmse_t, qmae_t, dmse_t, dmae_t], options=run_options, feed_dict={nn.keep_prob: args.keep_prob}, run_metadata=run_metadata)
        
        #update averages
        num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = update_averages(num_t, tmploss_avg_t, tmploss, emse_avg_t, emse, emae_avg_t, emae, fmse_avg_t, fmse, fmae_avg_t, fmae, qmse_avg_t, qmse, qmae_avg_t, qmae, dmse_avg_t, dmse, dmae_avg_t, dmae)

        #save progress
        if (step % args.save_interval == 0):
            saver.save(sess, step_checkpoint, global_step=step)   

        #check performance on the validation set
        if (step % args.validation_interval == 0):
            #save backup variables and load averaged variables
            sess.run(save_variable_backups_op)
            sess.run(load_averaged_variables_op)

            #initialize averages to 0
            num_v, tmploss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = reset_averages()
            #compute averages
            for i in range(args.num_valid//args.valid_batch_size):
                tmploss, emse, emae, fmse, fmae, qmse, qmae, dmse, dmae = sess.run([loss_v, emse_v, emae_v, fmse_v, fmae_v, qmse_v, qmae_v, dmse_v, dmae_v])
                num_v, tmploss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = update_averages(num_v, tmploss_avg_v, tmploss, emse_avg_v, emse, emae_avg_v, emae, fmse_avg_v, fmse, fmae_avg_v, fmae, qmse_avg_v, qmse, qmae_avg_v, qmae, dmse_avg_v, dmse, dmae_avg_v, dmae)

            #store results in dictionary
            results = {}
            results["loss_valid"] = tmploss_avg_v
            if data.E is not None:
                results["energy_mae_valid"]  = emae_avg_v
                results["energy_rmse_valid"] = np.sqrt(emse_avg_v)
            if data.F is not None:
                results["forces_mae_valid"]  = fmae_avg_v
                results["forces_rmse_valid"] = np.sqrt(fmse_avg_v)
            if data.Q is not None:
                results["charge_mae_valid"]  = qmae_avg_v
                results["charge_rmse_valid"] = np.sqrt(qmse_avg_v)
            if data.D is not None:
                results["dipole_mae_valid"]  = dmae_avg_v
                results["dipole_rmse_valid"] = np.sqrt(dmse_avg_v)

            if results["loss_valid"] < best_loss:
                best_loss   = results["loss_valid"]
                if data.E is not None:
                    best_emae   = results["energy_mae_valid"]
                    best_ermse  = results["energy_rmse_valid"]
                else:
                    best_emae  = np.Inf
                    best_ermse = np.Inf
                if data.F is not None:
                    best_fmae   = results["forces_mae_valid"]
                    best_frmse  = results["forces_rmse_valid"]
                else:
                    best_fmae  = np.Inf
                    best_frmse = np.Inf
                if data.Q is not None:
                    best_qmae   = results["charge_mae_valid"]
                    best_qrmse  = results["charge_rmse_valid"]
                else:
                    best_qmae  = np.Inf
                    best_qrmse = np.Inf
                if data.D is not None:
                    best_dmae   = results["dipole_mae_valid"]
                    best_drmse  = results["dipole_rmse_valid"]
                else:
                    best_dmae  = np.Inf
                    best_drmse = np.Inf
                best_step = step
                np.savez(best_loss_file, loss=best_loss, emae=best_emae,   ermse=best_ermse, 
                                         fmae=best_fmae,   frmse=best_frmse, 
                                         qmae=best_qmae,   qrmse=best_qrmse, 
                                         dmae=best_dmae,   drmse=best_drmse, 
                                         step=best_step)
                nn.save(sess, best_checkpoint, global_step=step)
            results["loss_best"] = best_loss
            if data.E is not None:
                results["energy_mae_best"]  = best_emae
                results["energy_rmse_best"] = best_ermse
            if data.F is not None:
                results["forces_mae_best"]  = best_fmae
                results["forces_rmse_best"] = best_frmse
            if data.Q is not None:
                results["charge_mae_best"]  = best_qmae
                results["charge_rmse_best"] = best_qrmse
            if data.D is not None:
                results["dipole_mae_best"]  = best_dmae
                results["dipole_rmse_best"] = best_drmse
            summary = create_summary(results)
            summary_writer.add_summary(summary, global_step=step)

            #restore backup variables
            sess.run(restore_variable_backups_op)
 
        #generate summaries
        if (step % args.summary_interval == 0) and (step > 0): 
            results = {}            
            results["loss_train"] = tmploss_avg_t
            if data.E is not None:
                results["energy_mae_train"]  = emae_avg_t
                results["energy_rmse_train"] = np.sqrt(emse_avg_t)
            if data.F is not None:
                results["forces_mae_train"]  = fmae_avg_t
                results["forces_rmse_train"] = np.sqrt(fmse_avg_t)
            if data.Q is not None:
                results["charge_mae_train"]  = qmae_avg_t
                results["charge_rmse_train"] = np.sqrt(qmse_avg_t)
            if data.D is not None:
                results["dipole_mae_train"]  = dmae_avg_t
                results["dipole_rmse_train"] = np.sqrt(dmse_avg_t)
            num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()

            summary = create_summary(results)
            summary_writer.add_summary(summary, global_step=step)
            nn_summary = nn_summary_op.eval()
            summary_writer.add_summary(nn_summary, global_step=step)
            if (args.record_run_metadata > 0):
                summary_writer.add_run_metadata(run_metadata, 'step %d' % step, global_step=step)
            if data.E is not None:
                print(str(step)+'/'+str(args.max_steps), "loss:", results["loss_train"], "best:", best_loss, "emae:", results["energy_mae_train"], "best:", best_emae)
