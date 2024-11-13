// SECTION 1 => `any`, `let`, `unions` and `enums`

// The `any` type. Just DONT use it. Instead use Interfaces
var item1: any = {id:1, name: "item1"};
item1 = {id: 2};

// Explicit casting
var item1 = <any>{id:1, name:"item1"};
item1 = {id:2};

var item1 = {id:1, name: "item1"} as any;
item1 = { id:2 };

// The let keyword. DONT use var
let index: number = 0;
if (index == 0) {
    let index: number = 2;
    console.log(`index = ${index}`);
}
console.log(`index = ${index}`);

// Const values
const constValue = "this should not be changed";
//constValue = "updated"; //ERROR

// Union types
function printObject(obj: string | number) {
    console.log(`obj = ${obj}`);
}
printObject(1);
printObject("string value");

// Type guards
function addWithTypeGuard(
    arg1: string | number,
    arg2: string | number
) {
    if (typeof arg1 === "string") {
        // arg1 is treated as a string
        console.log(`arg1 is of type string`);
        return arg1 + arg2;
    }
    if (typeof arg1 === "number" && typeof arg2 === "number"){
        // bboth are numbers
        console.log(`arg1 and arg2 are numbers`);
        return arg1 + arg2;
    }
    console.log(`default return treat both as strings`)
    return arg1.toString() + arg2.toString();
}
console.log(` "1", "2" = ${addWithTypeGuard("1","2")}`);
console.log(` 1, 2 = ${addWithTypeGuard(1,2)}`);
console.log(` 1, "2" = ${addWithTypeGuard(1,"2")}`);
console.log(` "1", 2 = ${addWithTypeGuard("1",2)}`);

// Type aliases
type StringOrNumber = string | number;
function addWithTypeAlias(
    arg1: StringOrNumber,
    arg2: StringOrNumber
){
    return arg1.toString() + arg2.toString();
}

//Enums
enum DoorState {
    Open = 3, // we are setting the value to 3 not 0 which is the default
    Closed = 7,
    Unspecified = 256
}
function checkDoorState(state: DoorState){
    console.log(`enum value is : ${state}`);
    switch (state){
        case DoorState.Open:
            console.log(`Door is open`);
            break;
        case DoorState.Closed:
            console.log(`Door is closed`);
            break;
    }
}
checkDoorState(DoorState.Open);
checkDoorState(DoorState.Closed);

// String enums
enum DoorStateString {
    OPEN = "Open",
    CLOSED = "Closed"
}
console.log(`OPEN = ${DoorStateString.OPEN}`);

// Const enums
const enum DoorStateConst {
    Open = 10,
    Closed = 20
}
console.log(`const Closed = ${DoorStateConst.Open}`);

// More primitive types
// Undefined
let array = ["123", "456", "789"];
delete array[0];
for (let i = 0; i < array.length; i++){
    console.log(`array[${i}] = ${array[i]}`);
}

for (let i = 0; i < array.length; i++){
    checkAndPrintElement(array[i]);
}
function checkAndPrintElement(arrElement: string | undefined){
    if (arrElement === undefined)
        console.log(`invalid array element`);
    else
        console.log(`valid array element ${arrElement}`);
}

// Null
function printValues(a: number|null){
    console.log(`a = ${a}`);
}
printValues(1);
printValues(null);

// Conditional expressions
const value : number = 10;
const message: string = value > 10 ? 
    "value is larger than 10" : "vakue is 10 or less";
console.log(message);

// Optional chaining
var objectA = {
    nestedProperty: {
        name: "nestedPropertyName"
    }
}
function printNestedObject(obj: any) {
    if (obj != undefined 
        && obj.nestedProperty != undefined
        && obj.nestedProperty.name
    ) {
        console.log(`name = ${obj.nestedProperty.name}`);
    } else {
        console.log(`name not found or undefined`);
    }
}
printNestedObject(objectA);

function printNestedOptionalChain(obj: any) {
    if (obj?.nestedProperty?.name){
        console.log(`name = ${obj.nestedProperty.name}`);
    } else {
        console.log(`name not found or undefined`);
    }
}
printNestedOptionalChain(undefined);
printNestedOptionalChain({
    aProperty: "another property"
});
printNestedOptionalChain({
    nestedProperty: {
        name: null
    }
});
printNestedOptionalChain({
    nestedProperty: {
        name: "nestedPropertyName"
    }
});

// Nullish coalescing
function nullishCheck(a: number | undefined | null){
    console.log(`a : ${a ?? `undefined or null`}`);
}
nullishCheck(1);
nullishCheck(null);
nullishCheck(undefined);

// Null or undefined operands
function testNullOperands(a: number, b: number | null | undefined){
    let addResult = a + (b ?? 0);
}

// Definite assignment
var globalString!: string; // use definite assignment assertion !
setGlobalString("this string is set");
console.log(`globalString = ${globalString}`);

function setGlobalString(value: string){
    globalString = value;
}

// Object
let structuredObject: object = {
    name: "myObject",
    properties: {
        id: 1,
        type: "AnObject"
    }
}
function printObjectType(a: object){
    console.log(`a: ${JSON.stringify(a)}`);
}
printObjectType(structuredObject);
//printObjectType("this is a string"); // Type checking produces error

// Unknown
let a: any = "test";
let aNumber: number = 2;
aNumber = a; // The flow of any logic

let u: unknown = "an unknown";
u = 1;
let aNumber2: number;
aNumber2 = <number>u;  // not allowed without explicit casting

// Never
function alwaysThrows(): never{
    throw new Error("this is always throw");
    //return -1;
}

// Never and switch
enum AnEnum {
    FIRST,
    SECOND
}
function getEnumValue(enumValue: AnEnum): string {
    switch (enumValue){
        case AnEnum.FIRST: return "First Case";
        case AnEnum.SECOND: return "Second Case";
    }
    let returnValue: never = enumValue;
    return returnValue;
}

// Object spread
let firstObj: object = {id:1, name: "firstObj"};
let secondObj: object = { ...firstObj};
console.log(`secondObj : ${JSON.stringify(secondObj)}`);

let nameObj: object = {name: "nameObj name"};
let idObj: object = {id:1};
let obj3 = {...nameObj, ...idObj};
console.log(`obj3 = ${JSON.stringify(obj3)}`);

// Spread precedence
let objPrec1: object = {id:1, name: "obj1 name"};
let objPrec2: object = {id:1001, desc: "obj2 description"};
let objPrec3: object = { ...objPrec1, ...objPrec2};
console.log(`objPrec3 : ${JSON.stringify(objPrec3, null,4)}`);

// Spread with arrays
let firstArray = [1,2,3];
let secondArray = [3,4,5];
let thirdArray = [...firstArray, ...secondArray];
console.log(`third array = ${thirdArray}`);

let objArray1 = [
    { id:1, name: "first element" },
]
let objArray2 = [
    { id:2, name: "second element" },
]
let objArray3 = [
    ...objArray1,
    { id:3, name: "third element" },
    ...objArray2
]
console.log(`objArray = ${JSON.stringify(objArray3, null,4)}`);

// Tuples
let tuple1: [string, boolean];
tuple1 = ["test", true];
// tuple1 = ["test"]; //error

// Tuple destructuring
console.log(`tuple1[0] : ${tuple1[0]}`);
console.log(`tuple1[1] : ${tuple1[1]}`);

let [tupleString, tupleBoolean] = tuple1;
console.log(`tupleString : ${tupleString}`);
console.log(`tupleBoolean : ${tupleBoolean}`);

// Optional tuple elements
let tupleOptional: [string, boolean?];
tupleOptional = ["test", true];
tupleOptional = ["tuple"];
console.log(`tupleOptional[0] : ${tupleOptional[0]}`);
console.log(`tupleOptional[1] : ${tupleOptional[1]}`);

// Tuples and spread syntax
let tupleRest: [number, ...string[]];
tupleRest = [1];
tupleRest = [1, "string1"];
tupleRest = [1, "string1", "string2"];

// Object destructuring
let complexObject = {
    aNum: 1,
    bStr: "name",
    cBool: true
}
let { aNum, bStr, cBool } = complexObject;
console.log(`aNum : ${aNum}`);
console.log(`bStr : ${bStr}`);
console.log(`cBool : ${cBool}`);
// rename variable names
let {aNum: objId, bStr: objName, cBool: isValid} = complexObject;
console.log(`objId : ${objId}`);
console.log(`objName : ${objName}`);
console.log(`isValid : ${isValid}`);

// FUNCTIONS

// Optional parameters
function concatValues(a: string, b?: string){
    console.log(`a + b = ${a + b}`);
}
concatValues("first", "second");
concatValues("third");

// Default parameters
function concatWithDefault(a: string, b: string="default"){
    console.log(`a + b = ${a + b}`);
}
concatWithDefault("first", "second");
concatWithDefault("third");

// Rest parameters
function testArguments(...args: string[] | number[]){
    for (let i in args){
        console.log(`args[${i}] = ${args[i]}`);
    }
}
testArguments("1");
testArguments(10,20);

// Function signatures as parameters
function myCallback(text: string): void {
    console.log(`myCallback called with ${text}`);
}
function withCallbackArg(
    message: string,
    callbackFn: (text: string) => void
){
    console.log(`withCallback called, message : ${message}`);
    callbackFn(`${message} from withCallback`);
}
withCallbackArg("initial text", myCallback);

// Function overrides
function add(a: string, b: string): string;
function add(a: number, b: number): number;
function add(a: boolean, b: boolean): boolean;
function add(a: any, b: any){
    return a + b;
}
console.log(`${add("first", "second")}`);
console.log(`${add(1,2)}`);
console.log(`${add(false,false)}`);

// Literals
type AllowedStringValues = "one" | "two" | "three";
type AllowedNumericValues = 1 | 20 | 65535;
function withLiteral(input: 
    AllowedStringValues | AllowedNumericValues
){
    console.log(`called with : ${input}`);
}
withLiteral("one");
withLiteral("two");
withLiteral("three");
withLiteral(65535);
//withLiteral("four");
//withLiteral(2);
