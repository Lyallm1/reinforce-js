export class Env {
  constructor(protected readonly width: number, protected readonly height: number, protected readonly numberOfStates: number, protected readonly numberOfActions: number) {}

  get(fieldname: string): number | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
}
