export class Opt {
  get(fieldname: string): any | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
}
