var myBoolean: boolean = true;
var myNumber: number = 1234;
var myStringArray: string[] = [`first`, `second`, `third`];

myBoolean = myNumber === 456;
myStringArray = [myNumber.toString(), `5678`];
myNumber = myStringArray.length;

console.log(`myBoolean = ${myBoolean}`);
console.log(`myStringArray = ${myStringArray}`);
console.log(`myNumber = ${myNumber}`);

// Inferred typing
var inferredString = "this is a string";
var inferredNumber = 1;

// Duck typing
var nameIdObject = { name: "myName", id: 1, print() { }};
nameIdObject = { id: 2, name: "anotherName",  print() { }};

// Duck typing ex2
var obj1 = {id: 1, print() { }};
var obj2 = {id: 2, print() { }, select() { }};
//obj1 = obj2;
//obj2 = obj1; //error

function calculate(a: number, b: number, c: number): number {
    return (a * b) + c;
}

console.log(`calculate() = ${calculate(2,3,1)}`);

/**
 * 
 * Given a string value, log it to the console
 * @param a     The input string
 */
function printString(a: string): void {
    console.log(a);
}

//var returnedValue : string = printString("this is a string"); // error

