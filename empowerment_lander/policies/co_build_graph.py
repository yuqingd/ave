import tensorflow as tf
import baselines.common.tf_util as U

def co_build_act(make_obs_ph, q_func, num_actions, scope="co_deepq", reuse=None, using_control_sharing=True):
  with tf.variable_scope(scope, reuse=reuse):
    observations_ph = make_obs_ph("observation")

    pilot_action_ph = tf.placeholder(tf.int32, (), name='pilot_action')
    pilot_tol_ph = tf.placeholder(tf.float32, (), name='pilot_tol')

    q_values = q_func(observations_ph.get(), num_actions, scope="q_func")

    batch_size = tf.shape(q_values)[0]

    q_values -= tf.reduce_min(q_values, axis=1) #normalize q values
    opt_actions = tf.argmax(q_values, axis=1, output_type=tf.int32)
    opt_q_values = tf.reduce_max(q_values, axis=1)

    batch_idxes = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    reshaped_batch_size = tf.reshape(batch_size, [1])

    pi_actions = tf.tile(tf.reshape(pilot_action_ph, [1]), reshaped_batch_size) #actions taken by pilot
    pi_act_idxes = tf.concat([batch_idxes, tf.reshape(pi_actions, [batch_size, 1])], axis=1)
    pi_act_q_values = tf.gather_nd(q_values, pi_act_idxes) #q values of actions taken by pilot

    # # if necessary, switch steering and keep main
    # mixed_actions = 3 * (pi_actions // 3) + (opt_actions % 3)
    # mixed_act_idxes = tf.concat([batch_idxes, tf.reshape(mixed_actions, [batch_size, 1])], axis=1)
    # mixed_act_q_values = tf.gather_nd(q_values, mixed_act_idxes)
    #
    # mixed_actions = tf.where(pi_act_q_values >= (1 - pilot_tol_ph) * mixed_act_q_values, pi_actions, mixed_actions)
    actions = tf.where(pi_act_q_values >= (1-pilot_tol_ph) * opt_q_values, pi_actions, opt_actions)

    act = U.function(inputs=[
        observations_ph, pilot_action_ph, pilot_tol_ph
      ],
                       outputs=[actions])
    return act


def co_build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=True, scope="deepq", reuse=None, using_control_sharing=True):

    act_f = co_build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse, using_control_sharing=using_control_sharing)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}
