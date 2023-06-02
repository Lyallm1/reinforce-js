import { Solver, TDEnv, TDOpt } from '../index.js';

import { Utils } from 'recurrent-js';

export class TDSolver extends Solver {
  protected readonly alpha: number;
  protected readonly epsilon: number;
  protected readonly gamma: number;
  protected readonly beta: number;
  protected readonly lambda: number;
  protected readonly numberOfPlanningSteps: number;
  protected readonly update: string;
  protected readonly qInitValue: number;
  protected readonly updateSmoothPolicy: boolean;
  protected readonly replacingTraces: boolean;
  protected numberOfActions: number;
  protected numberOfStates: number;
  protected pq: number[] | Float64Array;
  protected saSeen: number[];
  protected envModelR: number[] | Float64Array;
  protected envModelS: number[] | Float64Array;
  protected eligibilityTraces: number[] | Float64Array;
  protected randomPolicies: number[] | Float64Array;
  protected Q: number[] | Float64Array;
  protected explored: boolean;
  protected a1: number;
  protected s1: number;
  protected r0: number;
  protected a0: number;
  protected s0: number;

  constructor(protected env: TDEnv, opt: TDOpt) {
    super(env, opt);
    this.alpha = opt.get('alpha');
    this.epsilon = opt.get('epsilon');
    this.gamma = opt.get('gamma');
    this.update = opt.get('update');
    this.updateSmoothPolicy = opt.get('smoothPolicyUpdate');
    this.beta = opt.get('beta');
    this.lambda = opt.get('lambda');
    this.replacingTraces = opt.get('replacingTraces');
    this.qInitValue = opt.get('qInitVal');
    this.numberOfPlanningSteps = opt.get('numberOfPlanningSteps');
    this.Q = null;
    this.randomPolicies = null;
    this.eligibilityTraces = null;
    this.envModelS = null;
    this.envModelR = null;
    this.reset();
  }

  reset() {
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numerOfActions');
    this.Q = this.randomPolicies = this.eligibilityTraces = this.envModelS = this.envModelR = this.pq = Utils.zeros(this.numberOfStates * this.numberOfActions);
    if (this.qInitValue !== 0) Utils.fillConst(this.Q, this.qInitValue);
    Utils.fillConst(this.envModelS, -1);
    this.saSeen = [];
    for (let state = 0; state < this.numberOfStates; state++) {
      const allowedActions = this.env.allowedActions(state);
      for (const action of allowedActions) this.randomPolicies[action * this.numberOfStates + state] = 1 / allowedActions.length
    }
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  }

  decide(state: number): number {
    const allowedActions = this.env.allowedActions(state), actionIndex = this.epsilonGreedyActionPolicy(allowedActions, Array.from(allowedActions, action => this.randomPolicies[action * this.numberOfStates + state]));
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = state;
    this.a1 = actionIndex;
    return actionIndex;
  }

  private epsilonGreedyActionPolicy(poss: number[], probs: number[]) {
    this.explored = Math.random() < this.epsilon ? true : false;
    return poss[Math.random() < this.epsilon ? Utils.randi(0, poss.length) : Utils.sampleWeighted(probs)];
  }

  learn(r1: number) {
    if (this.r0 !== null) {
      this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
      if (this.numberOfPlanningSteps > 0) {
        const sa = this.a0 * this.numberOfStates + this.s0;
        if (this.envModelS[sa] === -1) this.saSeen.push(this.a0 * this.numberOfStates + this.s0);
        this.envModelS[sa] = this.s1;
        this.envModelR[sa] = this.r0;
        const spq = [];
        for (const sa of this.saSeen) {
          const sap = this.pq[sa];
          if (sap > 1e-5) spq.push({ 'sa': sa, 'p': sap });
        }
        for (let k = 0; k < Math.min(this.numberOfPlanningSteps, spq.sort((a, b) => a.p < b.p ? 1 : -1).length); k++) {
          const s0a0 = spq[k].sa;
          this.pq[s0a0] = 0;
          this.learnFromTuple(s0a0 % this.numberOfStates, Math.floor(s0a0 / this.numberOfStates), this.envModelR[s0a0], this.envModelS[s0a0], -1, 0);
        }
      }
    }
    this.r0 = r1;
  }

  private learnFromTuple(s0: number, a0: number, r0: number, s1: number, a1: number, lambda: number) {
    const sa = a0 * this.numberOfStates + s0;
    let target: number;
    if (this.update === 'qlearn') {
      const poss = this.env.allowedActions(s1);
      let qmax = 0;
      poss.forEach((p, i) => {
        const qval = this.Q[p * this.numberOfStates + s1];
        if (i === 0 || qval > qmax) qmax = qval;
      });
      target = r0 + this.gamma * qmax;
    } else if (this.update === 'sarsa') target = r0 + this.gamma * this.Q[a1 * this.numberOfStates + s1];
    if (lambda > 0) {
      this.eligibilityTraces[sa] = null ? 1 : this.eligibilityTraces[sa] + 1;
      const decay = lambda * this.gamma, stateUpdate = Utils.zeros(this.numberOfStates);
      for (let s = 0; s < this.numberOfStates; s++) {
        for (const a of this.env.allowedActions(s)) {
          const saloop = a * this.numberOfStates + s;
          const esa = this.eligibilityTraces[saloop];
          const update = this.alpha * esa * (target - this.Q[saloop]);
          this.Q[saloop] += update;
          this.updatePriority(s, a, update);
          this.eligibilityTraces[saloop] *= decay;
          const u = Math.abs(update);
          if (u > stateUpdate[s]) stateUpdate[s] = u;
        }
        if (stateUpdate[s] > 1e-5) this.updatePolicy(s);
      }
      if (this.explored && this.update === 'qlearn') this.eligibilityTraces = Utils.zeros(this.numberOfStates * this.numberOfActions);
    } else {
      const update = this.alpha * (target - this.Q[sa]);
      this.Q[sa] += update;
      this.updatePriority(s0, a0, update);
      this.updatePolicy(s0);
    }
  }

  private updatePriority(s: number, a: number, u: number) {
    u = Math.abs(u);
    if (u < 1e-5 || this.numberOfPlanningSteps === 0) return;
    for (let si = 0; si < this.numberOfStates; si++) for (let ai = 0; ai < this.numberOfActions; ai++) {
      const siai = ai * this.numberOfStates + si;
      if (this.envModelS[siai] === s) this.pq[siai] += u;
    }
  }

  private updatePolicy(s: number) {
    let qmax: number, nmax: number, psum = 0;
    const poss = this.env.allowedActions(s), qs = [];
    poss.forEach((a, i) => {
      const qval = this.Q[a * this.numberOfStates + s];
      qs.push(qval);
      if (i === 0 || qval > qmax) [qmax, nmax] = [qval, 1];
      else if (qval === qmax) nmax++
      const target = qs[i] === qmax ? 1 / nmax : 0, ix = a * this.numberOfStates + s;
      if (this.updateSmoothPolicy) {
        this.randomPolicies[ix] += this.beta * (target - this.randomPolicies[ix]);
        psum += this.randomPolicies[ix];
      } else this.randomPolicies[ix] = target;
    });
    if (this.updateSmoothPolicy) for (const a of poss) this.randomPolicies[a * this.numberOfStates + s] /= psum;
  }

  toJSON(): object {
    throw new Error('Not implemented yet.');
  }

  fromJSON(json: {}) {
    throw new Error('Not implemented yet.');
  }
}
