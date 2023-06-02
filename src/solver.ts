import { Env, Opt } from './index.js';

export abstract class Solver {
  constructor(protected env: Env, protected opt: Opt) {}

  getOpt() {
    return this.opt;
  }

  getEnv() {
    return this.env;
  }

  abstract decide(stateList: any): number;
  abstract learn(r1: number): void;
  abstract reset(): void;
  abstract toJSON(): object;
  abstract fromJSON(json: {}): void;
}
