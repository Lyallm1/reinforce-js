import { Env } from '../index.js';

export class TDEnv extends Env {
  allowedActions(s: number): number[] {
    const x = this.stox(s), y = this.stoy(s), allowedActions: number[] = [];
    if (x > 0) allowedActions.push(0);
    if (y > 0) allowedActions.push(1);
    if (y < this.height - 1) allowedActions.push(2);
    if (x < this.width - 1) allowedActions.push(3);
    return allowedActions;
  }

  protected stox(s: number): number {
    return Math.floor(s / this.height);
  }

  protected stoy(s: number): number {
    return s % this.height;
  }
}
