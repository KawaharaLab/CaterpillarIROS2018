#![crate_type = "dylib"]

extern crate libc;
use std::slice;


const SOMITES_AMOUNT: usize = 5;

#[repr(C)]
#[derive(Default)]
pub struct CSomite {
    position: f64,
    verocity: f64,
    force: f64,
    mass: f64,
    friction_coeff: f64,
}

impl CSomite {
    fn update_position(&mut self, dt: f64) {
        self.position = self.position + self.verocity*dt + 0.5*dt*dt*self.force/self.mass;
    }

    fn clone_from(&mut self, source: &Self) {
        self.position = source.position;
        self.verocity = source.verocity;
        self.force = source.force;
        self.mass = source.mass;
        self.friction_coeff = source.friction_coeff;
    }
}

#[repr(C)]
#[derive(Default)]
pub struct CRTS {
    position_0: f64,
    position_1: f64,
    verocity_0: f64,
    verocity_1: f64,
    phase: f64,
    natural_length: f64,
    angular_verocity: f64,
    max_length: f64,
    amplitude: f64,
    spring_const: f64,
    dump_coeff: f64,
}

impl CRTS {
    fn clone_from(&mut self, source: &Self) {
        self.position_0 = source.position_0;
        self.position_1 = source.position_1;
        self.verocity_0 = source.verocity_0;
        self.verocity_1 = source.verocity_1;
        self.phase = source.phase;
        self.natural_length = source.natural_length;
        self.angular_verocity = source.angular_verocity;
        self.max_length = source.max_length;
        self.amplitude = source.amplitude;
        self.spring_const = source.spring_const;
        self.dump_coeff = source.dump_coeff;
    }

    fn update_phase(&mut self, time_delta: f64) {
        self.phase += self.angular_verocity * time_delta;
    }

    fn update_natural_length(&mut self) {
        let expand_rate = 1. + self.amplitude * (self.phase.cos() - 1.);
        self.natural_length = self.max_length * expand_rate;
    }

    fn calculate_force(&self) -> f64 {
        let length = (self.position_1 - self.position_0).abs();
        let mut force = - self.dump_coeff * (self.verocity_1 - self.verocity_0);
        if length > 0.0001 {
            force -= self.spring_const * (length - self.natural_length);
        }
        force
    }
}

#[repr(C)]
pub struct CCaterpillarState {
    somites: [CSomite; SOMITES_AMOUNT],
    rtses: [CRTS; SOMITES_AMOUNT - 1],
    frictions: [f64; SOMITES_AMOUNT],
}

fn mean(a: f64, b: f64) -> f64 {
    (a + b) / 2.0
}

#[no_mangle]
pub extern fn calculate_force(
        pos0_x: f64, pos1_x: f64, v0_x: f64, v1_x: f64,
        natural_length: f64, spring_const: f64, dump_coeff: f64) -> f64 {
    let length = (pos1_x - pos0_x).abs();
    let mut force = 0.;

    if length > 0.0001 {
        force = - spring_const * (length - natural_length);
    }

    force -= dump_coeff * (v1_x - v0_x);
    force
}

// #[no_mangle]
// pub extern fn release_caterpillar_state() {
//
// }

#[no_mangle]
pub extern fn update_caterpillar(somites_ptr: *const CSomite, rts_ptr: *const CRTS, time_delta: f64) -> CCaterpillarState{
    let somites = unsafe { slice::from_raw_parts(somites_ptr, SOMITES_AMOUNT) };
    let rtses = unsafe { slice::from_raw_parts(rts_ptr, SOMITES_AMOUNT - 1) };

    // Convert to array for returning
    let mut new_somites: [CSomite; SOMITES_AMOUNT] = Default::default();
    for i in 0..new_somites.len() {
        new_somites[i].clone_from(&somites[i]);
    }

    let mut new_rtses: [CRTS; SOMITES_AMOUNT - 1] = Default::default();
    for i in 0..new_rtses.len() {
        new_rtses[i].clone_from(&rtses[i]);
    }

    // Update somites postions
    for somite in new_somites.iter_mut() {
        somite.update_position(time_delta);
    }

    let mut forces = [0.0; SOMITES_AMOUNT];
    let mut frictions = [0.0; SOMITES_AMOUNT];

    // Calculate friction for each somites
    for i in 0..new_somites.len() {
        let mu = if i == 0 {
                (new_somites[i + 1].position - new_somites[i].position).abs() * 10.0
            } else if i == somites.len() - 1 {
                (new_somites[i].position - new_somites[i - 1].position).abs() * 10.0
            } else {
                mean((new_somites[i].position - new_somites[i - 1].position).abs(), (new_somites[i + 1].position - new_somites[i].position).abs()) * 10.0
            };
        forces[i] -= mu * new_somites[i].verocity * new_somites[i].friction_coeff;
        frictions[i] = - mu * new_somites[i].verocity * new_somites[i].friction_coeff;
    }

    // Update rts
    for i in 0..new_rtses.len() {
        new_rtses[i].update_phase(time_delta);
        new_rtses[i].update_natural_length();
        let rts_force = new_rtses[i].calculate_force();
        // Apply rtses force on somites
        forces[i] -= rts_force;
        forces[i + 1] += rts_force;
    }

    // Update verocities
    for i in 0..new_somites.len() {
        new_somites[i].verocity += mean(new_somites[i].force, forces[i]) / new_somites[i].mass * time_delta;
    }

    // Update forces
    for i in 0..new_somites.len() {
        new_somites[i].force = forces[i];
    }

    CCaterpillarState{
        somites: new_somites,
        rtses: new_rtses,
        frictions: frictions,
    }
}
