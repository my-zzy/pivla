"""Separate state-driven demonstration expert; never used by learned evaluation."""
import numpy as np
from .expert_policy import ScriptedExpert


class RecoveryExpert(ScriptedExpert):
    """Live block alignment, short settling dwell, and bounded physical retries."""
    def __init__(self, config):
        super().__init__(config)
        self.retries = 0
        self.lost_contact_s = 0.
        self.recovery_xy = None
        self.initialized = False
        self.lift_xy = None

    def transition(self, stage, t, obs, reason):
        self.events.append({'stage':self.stage,'next_stage':stage,'end_time_s':t,
                            'duration_s':t-self.entered,'reason':reason})
        self.stage, self.entered, self.dwell = stage, t, 0.
        self.lost_contact_s = 0.
        if stage == 'lift':
            # A grasped block moves with the drone. Following its live XY during
            # lifting removes the position-restoring target and can sustain drift.
            self.lift_xy = np.array(obs['block_state'][:2]).copy()
        print(f'Recovery expert {t:.2f}s -> {stage}: {reason}',flush=True)

    def recover(self, t, obs, reason):
        self.retries += 1
        if self.retries > 3:
            raise RuntimeError('Recovery retry budget exhausted')
        self.recovery_xy = np.array(obs['drone_state'][:2])
        # Discard the stale low target before retracting vertically.
        self.target = np.array(obs['drone_state'][:3])
        self.transition('recover',t,obs,reason)

    def update(self, t, obs):
        dt = max(0.,t-self.last_time); self.last_time=t
        drone,block=np.array(obs['drone_state']),np.array(obs['block_state'])
        if not self.initialized:
            self.target=drone[:3].copy(); self.initialized=True
        if t-self.entered>self.task['stage_timeout_s']:
            raise RuntimeError(f'Recovery expert stage {self.stage} timed out')
        slow=np.linalg.norm(drone[7:10])<.035
        qw,qx,qy,qz=drone[3:7]
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        yaw_error=abs((yaw-self.cfg['target']['yaw']+np.pi)%(2*np.pi)-np.pi)
        heading_settled=yaw_error<.03 and np.linalg.norm(drone[10:13])<.05
        contacts=all(obs['contacts'])
        if self.stage in ('lift','transport'):
            self.lost_contact_s=0. if contacts else self.lost_contact_s+dt
            if self.lost_contact_s>=.3:
                self.recover(t,obs,'lost contact'); dt=0.
        xy_error=np.linalg.norm(drone[:2]-block[:2])
        if self.stage=='grasp' and (xy_error>.04 or (t-self.entered>2. and not contacts)):
            self.recover(t,obs,'missed closure or alignment'); dt=0.
        if self.stage in ('approach','descend','grasp','recover'):
            if block[2] < self.task['table_top_m']-.02:
                raise RuntimeError('Block left the table; do not collect unsafe recovery')
        goal=np.array([*block[:2],self.task['flight_height_m']])
        if self.stage in ('descend','grasp'):
            goal[2]=self.task['grasp_height_m']
        elif self.stage=='lift' and self.lift_xy is not None:
            goal[:2]=self.lift_xy
        elif self.stage=='recover':
            goal[:2]=self.recovery_xy
        elif self.stage in ('transport','release','retreat'):
            goal[:2]=self.task['bin_position'][:2]
            if self.stage=='retreat':goal[1]-=.5
        delta=goal-self.target
        self.target += delta*min(1.,self.task['setpoint_speed_m_s']*dt/max(np.linalg.norm(delta),1e-9))
        error=np.linalg.norm(drone[:3]-goal)
        next_stage=None
        if self.stage in ('approach','recover'):
            condition=error<.015 and slow and heading_settled and obs['aperture_m']>.12
            next_stage='descend' if self.stage=='approach' else 'approach'
        elif self.stage=='descend':
            condition=error<.012 and slow and heading_settled and xy_error<.012 and obs['aperture_m']>.12
            next_stage='grasp'
        elif self.stage=='grasp':
            condition=contacts and .07<obs['aperture_m']<.095 and slow
            next_stage='lift'
        elif self.stage=='lift':
            condition=contacts and block[2]>self.task['table_top_m']+.30 and error<self.task['loaded_position_tolerance_m'] and slow
            next_stage='transport'
        elif self.stage=='transport':
            radius=np.linalg.norm(self.task['block_size_m'])/2
            clearance=np.array(self.task['bin_inner_size_m'])/2-np.abs(block[:2]-goal[:2])-radius
            condition=contacts and np.all(clearance>self.task.get('release_footprint_margin_m',.05)) and slow and block[2]>.80
            next_stage='release'
        elif self.stage=='release':
            condition=self.in_bin(obs) and not any(obs['contacts']) and obs['aperture_m']>.12
            next_stage='retreat'
        else:
            condition=self.in_bin(obs) and error<.025 and slow
        self.dwell=self.dwell+dt if condition else 0.
        # Settling/debounce only; no measurement-only minimum stage age.
        threshold=2. if self.stage=='release' else .5
        if self.dwell>=threshold:
            if self.stage=='retreat':self.done=True
            else:self.transition(next_stage,t,obs,'physical condition settled')
        return self.target.tolist(),self.stage in ('approach','descend','recover','release','retreat')
