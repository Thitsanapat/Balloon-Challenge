"""Infer actuator histories from a public state replay with the official 6-DoF ODE.

This is offline system-identification tooling, not an evaluation agent.  It
uses the simulator's derivative function to recover a smooth, rate-limited
control sequence whose state derivatives match a recorded feasible flight.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def build_flight(cache_path, seed, launch_time, launch_attitude):
    scenario, _ = load_scenario_parameters(1)
    scenario = copy.deepcopy(scenario)
    scenario['scenario']['random_seed'] = seed
    with np.load(cache_path) as cache:
        flights = np.asarray(cache['balloon_flights'])
        release_steps = np.asarray(cache['release_steps'], dtype=int)
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    env._BalloonPoppingEnv__generate_balloon_flights = lambda: setattr(
        env, '_balloon_flights', flights,
    )
    env.reset(seed=seed)
    dt = float(scenario['simulation']['time_step'])
    launch_step = int(round(launch_time/dt))
    env.current_step = launch_step-1
    env._balloon_states = flights[:, :, env.current_step].copy()
    env._balloon_status[:, 0] = (
        env.current_step >= release_steps
    ).astype(env._balloon_status.dtype)
    env.step({
        'launch': True,
        'launch_inclination_heading': np.asarray(launch_attitude, dtype=float),
        'tvc': np.zeros(2),
        'roll': 0.,
        'throttle': 1.,
    })
    return env._rocket_flight, dt


def load_reference(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    records = payload['trajectories']
    valid = [record for record in records if record['rocket_states'][0] is not None]
    times = np.asarray([record['time'] for record in valid], dtype=float)
    states = np.asarray([record['rocket_states'] for record in valid], dtype=float)
    return times, states


def set_raw_controls(flight, controls):
    tvc = flight.rocket.thrust_vector_control
    tvc.x._actuator_output = float(controls[0])
    tvc.y._actuator_output = float(controls[1])
    flight.rocket.throttle_control._actuator_output = float(controls[2])
    flight.rocket.roll_control._actuator_output = float(controls[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--launch-time', type=float, required=True)
    parser.add_argument('--launch-attitude', type=float, nargs=2, required=True)
    parser.add_argument('--window', type=int, default=9)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()

    times, states = load_reference(args.replay)
    if args.max_samples is not None:
        times, states = times[:args.max_samples], states[:args.max_samples]
    window = min(args.window, len(times) if len(times) % 2 else len(times)-1)
    if window < 5:
        raise ValueError('reference needs at least five samples')
    acceleration = savgol_filter(
        states[:, 3:6], window, 3, deriv=1,
        delta=float(np.median(np.diff(times))), axis=0,
    )
    angular_acceleration = savgol_filter(
        states[:, 10:13], window, 3, deriv=1,
        delta=float(np.median(np.diff(times))), axis=0,
    )
    flight, dt = build_flight(
        args.cache, args.seed, args.launch_time, args.launch_attitude,
    )

    controls = np.empty((len(times), 4))
    residuals = np.empty((len(times), 6))
    previous = np.array([0., 0., 1., 0.])
    absolute_low = np.array([-15., -15., 0., -10.])
    absolute_high = np.array([15., 15., 1., 10.])
    max_step = np.array([60.*dt, 60.*dt, 2.*dt, 20.*dt])

    for index, (time, state) in enumerate(zip(times, states)):
        desired = np.r_[acceleration[index], angular_acceleration[index]]
        low = np.maximum(absolute_low, previous-max_step)
        high = np.minimum(absolute_high, previous+max_step)

        def error(candidate):
            set_raw_controls(flight, candidate)
            derivative = np.asarray(flight.u_dot_generalized(time, state), dtype=float)
            # Translational and rotational derivatives have comparable impact
            # after scaling; angular residuals otherwise dominate numerically.
            return np.r_[(derivative[3:6]-desired[:3])/2.,
                         (derivative[10:13]-desired[3:])/2.]

        result = least_squares(
            error, np.clip(previous, low, high), bounds=(low, high),
            max_nfev=12, ftol=1e-7, xtol=1e-7, gtol=1e-7,
        )
        previous = result.x
        controls[index] = previous
        residuals[index] = 2.*error(previous)
        if index % 250 == 0:
            print(
                f'{index}/{len(times)} t={time:.2f} '
                f'control={np.round(previous, 3)} '
                f'rms={np.sqrt(np.mean(residuals[index]**2)):.3f}',
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        times=times,
        states=states,
        controls=controls,
        derivative_residuals=residuals,
        seed=args.seed,
        launch_time=args.launch_time,
        launch_attitude=np.asarray(args.launch_attitude),
    )
    print(
        f'saved {args.output}; residual RMS '
        f'{np.sqrt(np.mean(residuals**2)):.4f}',
    )


if __name__ == '__main__':
    main()
