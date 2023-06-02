import { DQNOpt, Env, Solver } from '../index.js';
import { Graph, Mat, Net, Utils } from 'recurrent-js';

import { SarsaExperience } from './sarsa.js';

export class DQNSolver extends Solver {
  numberOfStates: number;
  numberOfActions: number;
  numberOfHiddenUnits: number[];
  readonly epsilonMax: number;
  readonly epsilonMin: number;
  readonly epsilonDecayPeriod: number;
  readonly epsilon: number;
  readonly gamma: number;
  readonly alpha: number;
  readonly doLossClipping: boolean;
  readonly lossClamp: number;
  readonly doRewardClipping: any;
  readonly rewardClamp: any;
  readonly experienceSize: number;
  readonly keepExperienceInterval: number;
  readonly replaySteps: number;
  protected net: Net;
  protected previousGraph: Graph;
  protected shortTermMemory: SarsaExperience = { s0: null, a0: null, r0: null, s1: null, a1: null };
  protected longTermMemory: SarsaExperience[];
  protected isInTrainingMode: boolean;
  protected learnTick: number;
  protected memoryIndexTick: number;

  constructor(env: Env, opt: DQNOpt) {
    super(env, opt);
    this.numberOfHiddenUnits = opt.get('numberOfHiddenUnits');
    this.epsilonMax = opt.get('epsilonMax');
    this.epsilonMin = opt.get('epsilonMin');
    this.epsilonDecayPeriod = opt.get('epsilonDecayPeriod');
    this.epsilon = opt.get('epsilon');
    this.experienceSize = opt.get('experienceSize');
    this.gamma = opt.get('gamma');
    this.alpha = opt.get('alpha');
    this.doLossClipping = opt.get('doLossClipping');
    this.lossClamp = opt.get('lossClamp');
    this.doRewardClipping = opt.get('doRewardClipping');
    this.rewardClamp = opt.get('rewardClamp');
    this.keepExperienceInterval = opt.get('keepExperienceInterval');
    this.replaySteps = opt.get('replaySteps');
    this.isInTrainingMode = opt.get('trainingMode');
    this.reset();
  }

  reset() {
    this.numberOfHiddenUnits = this.opt.get('numberOfHiddenUnits');
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numberOfActions');
    this.net = new Net({ architecture: { inputSize: this.numberOfStates, hiddenUnits: this.numberOfHiddenUnits, outputSize: this.numberOfActions } });
    this.learnTick = this.memoryIndexTick = 0;
    this.shortTermMemory.s0 = null;
    this.shortTermMemory.a0 = null;
    this.shortTermMemory.r0 = null;
    this.shortTermMemory.s1 = null;
    this.shortTermMemory.a1 = null;
    this.longTermMemory = [];
  }
  
  setTrainingModeTo(trainingMode: boolean) {
    this.isInTrainingMode = trainingMode;
  }

  getTrainingMode(): boolean {
    return this.isInTrainingMode;
  }

  toJSON() {
    return { ns: this.numberOfStates, nh: this.numberOfHiddenUnits, na: this.numberOfActions, net: Net.toJSON(this.net) };
  }

  fromJSON(json: { ns: number, nh: number[], na: number, net: Net }) {
    this.numberOfStates = json.ns;
    this.numberOfHiddenUnits = json.nh;
    this.numberOfActions = json.na;
    this.net = Net.fromJSON(json.net);
  }

  decide(state: Array<number>): number {
    const stateVector = new Mat(this.numberOfStates, 1);
    stateVector.setFrom(state);
    const actionIndex = this.epsilonGreedyActionPolicy(stateVector);
    this.shiftStateMemory(stateVector, actionIndex);
    return actionIndex;
  }

  protected epsilonGreedyActionPolicy(stateVector: Mat): number {
    return Math.random() < this.currentEpsilon() ? Utils.argmax(this.forwardQ(stateVector).w) : Utils.randi(0, this.numberOfActions);
  }

  protected currentEpsilon(): number {
    return this.isInTrainingMode ? (this.learnTick < this.epsilonDecayPeriod ? this.epsilonMax - (this.epsilonMax - this.epsilonMin) / this.epsilonDecayPeriod * this.learnTick : this.epsilonMin) : this.epsilon;
  }

  protected forwardQ(stateVector: Mat | null): Mat {
    return this.determineActionVector(new Graph(), stateVector);
  }

  protected backwardQ(stateVector: Mat | null): Mat {
    const graph = new Graph();
    graph.memorizeOperationSequence(true);
    return this.determineActionVector(graph, stateVector);
  }


  protected determineActionVector(graph: Graph, stateVector: Mat): Mat {
    const a2mat = this.net.forward(stateVector, graph);
    this.backupGraph(graph);
    return a2mat;
  }

  protected backupGraph(graph: Graph) {
    this.previousGraph = graph;
  }

  protected shiftStateMemory(stateVector: Mat, actionIndex: number) {
    this.shortTermMemory.s0 = this.shortTermMemory.s1;
    this.shortTermMemory.a0 = this.shortTermMemory.a1;
    this.shortTermMemory.s1 = stateVector;
    this.shortTermMemory.a1 = actionIndex;
  }

  learn(r: number) {
    if (this.shortTermMemory.r0 && this.alpha > 0) {
      this.learnFromSarsaTuple(this.shortTermMemory);
      this.addToReplayMemory();
      this.limitedSampledReplayLearning();
    }
    this.shiftRewardIntoMemory(r);
  }

  private shiftRewardIntoMemory(r: number) {
    this.shortTermMemory.r0 = this.clipReward(r);
  }

  protected clipReward(r: number): number {
    return this.doRewardClipping ? Math.sign(r) * Math.min(Math.abs(r), this.rewardClamp) : r;
  }

  protected learnFromSarsaTuple(sarsa: SarsaExperience) {
    const q0ActionVector = this.backwardQ(sarsa.s0);
    q0ActionVector.dw[sarsa.a0] = this.clipLoss(q0ActionVector.w[sarsa.a0] - this.getTargetQ(sarsa.s1, sarsa.r0));
    this.previousGraph.backward();
    this.net.update(this.alpha);
  }

  protected getTargetQ(s1: Mat, r0: number): number {
    const targetActionVector = this.forwardQ(s1);
    return r0 + this.gamma * targetActionVector.w[Utils.argmax(targetActionVector.w)];
  }

  protected clipLoss(loss: number): number {
    if (this.doLossClipping) {
      if (loss > this.lossClamp) loss = this.lossClamp;
      else if (loss < -this.lossClamp) loss = -this.lossClamp;
    }
    return loss;
  }

  protected addToReplayMemory() {
    if (this.learnTick % this.keepExperienceInterval === 0) this.addShortTermToLongTermMemory();
    this.learnTick++;
  }

  protected addShortTermToLongTermMemory() {
    this.longTermMemory[this.memoryIndexTick] = this.extractSarsaExperience();
    this.memoryIndexTick++;
    if (this.memoryIndexTick > this.experienceSize - 1) this.memoryIndexTick = 0;
  }

  protected extractSarsaExperience(): SarsaExperience {
    const s0 = new Mat(this.shortTermMemory.s0.rows, this.shortTermMemory.s0.cols);
    s0.setFrom(this.shortTermMemory.s0.w);
    const s1 = new Mat(this.shortTermMemory.s1.rows, this.shortTermMemory.s1.cols);
    s1.setFrom(this.shortTermMemory.s1.w);
    return { s0, a0: this.shortTermMemory.a0, r0: this.shortTermMemory.r0, s1, a1: this.shortTermMemory.a1 };
  }

  protected limitedSampledReplayLearning() {
    for (let i = 0; i < this.replaySteps; i++) this.learnFromSarsaTuple(this.longTermMemory[Utils.randi(0, this.longTermMemory.length)]);
  }
}
