import { Opt } from '../index.js';

export class DQNOpt extends Opt {
  protected trainingMode = true;
  protected numberOfHiddenUnits = [100];
  protected epsilonMax = 1;
  protected epsilonMin = 0.1;
  protected epsilonDecayPeriod = 1000000;
  protected epsilon = 0.05;
  protected gamma = 0.9;
  protected alpha = 0.01;
  protected experienceSize = 1000000;
  protected doLossClipping = true;
  protected lossClamp = 1;
  protected doRewardClipping = true;
  protected rewardClamp = 1;
  protected keepExperienceInterval = 25;
  protected replaySteps = 10;

  setNumberOfHiddenUnits(numberOfHiddenUnits: number[]) {
    this.numberOfHiddenUnits = numberOfHiddenUnits;
  }

  setEpsilonDecay(epsilonMax: number, epsilonMin: number, epsilonDecayPeriod: number) {
    this.epsilonMax = epsilonMax;
    this.epsilonMin = epsilonMin;
    this.epsilonDecayPeriod = epsilonDecayPeriod;
  }

  setEpsilon(epsilon: number) {
    this.epsilon = epsilon;
  }

  setGamma(gamma: number) {
    this.gamma = gamma;
  }

  setAlpha(alpha: number) {
    this.alpha = alpha;
  }

  setLossClipping(doLossClipping: boolean) {
    this.doLossClipping = doLossClipping;
  }

  setLossClamp(lossClamp: number) {
    this.lossClamp = lossClamp;
  }

  setRewardClipping(doRewardClipping: boolean) {
    this.doRewardClipping = doRewardClipping;
  }

  setRewardClamp(rewardClamp: number) {
    this.rewardClamp = rewardClamp;
  }

  setTrainingMode(trainingMode: boolean) {
    this.trainingMode = trainingMode;
  }

  setExperienceSize(experienceSize: number) {
    this.experienceSize = experienceSize;
  }

  setReplayInterval(keepExperienceInterval: number) {
    this.keepExperienceInterval = keepExperienceInterval;
  }

  setReplaySteps(replaySteps: number) {
    this.replaySteps = replaySteps;
  }
}
