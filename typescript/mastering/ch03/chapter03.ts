// CHAPTER 03 Interfaces, Classes, Inheritance and Modules.

// INTERFACES

interface IIdName{
    id: number;
    name: string;
}

let idObject: IIdName = {
    id: 2,
    name: "this is a name"
}

// Optional properties
interface IOptional {
    id: number;
    name?: string;
}
let optionalId: IOptional = {
    id: 1
}
let optionalIdName: IOptional = {
    id:2,
    name: "optional name"
}

// Weak types
interface IWeakType {
    id?: number,
    name?: string
}

// The in operator
interface IIdName {
    id: number;
    name: string;
}
interface IDescrValue {
    descr: string;
    value: number;
}
function printNameOrValue(
    obj: IIdName | IDescrValue
): void {
        if ('id' in obj){
            console.log(`obj.name : ${obj.name}`);
        }
        if ('descr' in obj){
            console.log(`obj.value : ${obj.value}`);
        }
}
printNameOrValue({
    id:1,
    name: "nameValue"
});
printNameOrValue({
    descr:"description",
    value: 2
});

// Keyof
interface IPerson {
    id: number;
    name: string;
}
type PersonPropertyName = keyof IPerson; 
function getProperty(key: PersonPropertyName, value: IPerson){
    console.log(`${key} = ${value[key]}`);
}
getProperty("id", {id:1, name: "firstName"});
getProperty("name", {id:2, name: "secondName"});
//getProperty("telephone", {id:3, name: "thirdName"}); // Error


//Classes and the `this` keyword

class SimpleClass {
    id: number | undefined;
    print(): void {
        console.log(`SimpleClass.print() called.`);
        console.log(`SimpleClass.id = ${this.id}`);
    }
}
let mySimpleCLass = new SimpleClass();
mySimpleCLass.id = 2020;
mySimpleCLass.print();


// Implementing interfaces
class ClassA implements IPrint{
    print(): void {
        console.log(`ClassA.print() called.`)
    };
}
class ClassB implements IPrint{
    print(): void {
        console.log(`ClassB.print() called.`)
    };
}
interface IPrint {
    print(): void;
}
function printClass(a: IPrint){
    a.print();
}
let classA = new ClassA();
let classB = new ClassB();
printClass(classA);
printClass(classB);

// Class constructors
class ClassWithConstructor {
    id: number;
    constructor(_id: number){
        this.id = _id;
    }
}
let classWithConstructor = new ClassWithConstructor(10);
console.log(`classWithConstructor = 
    ${JSON.stringify(classWithConstructor)}`);

// Class modifiers
class ClassWithPublicProperty {
    public id: number | undefined;
}
let publicAccess = new ClassWithPublicProperty();
publicAccess.id = 10;

class ClassWithPrivateProperty {
    private id: number;
    constructor(id: number){
        this.id = id;
    }
}
let privateAccess = new ClassWithPrivateProperty(10);
//privateAccess.id = 20;

// Javascript private fields Node v12 and later
class ClassES6Private {
    #id: number;
    constructor(id: number){
        this.#id = id;
    }
}
let es6PrivateClass = new ClassES6Private(10);
//es6PrivateClass.#id = 20;

//Constructor parameter properties
class ClassWithCtorMods{
    constructor(public id: number, private name: string){

    }
}
let myClassMod = new ClassWithCtorMods(1, "test");
console.log(`myClassMod.id = ${myClassMod.id}`);
//console.log(`myClassMod.name = ${myClassMod.name}`);

// Readonly
class ClassWithReadonly {
    readonly name: string;
    constructor(_name: string){
        this.name = _name;
    }
    setNameValue(_name: string){
        //this.name = _name;
    }
}

// Get and Set
class ClassWithAccessors{
    private _id: number = 0;
    get id(): number{
        console.log(`get id property`);
        return this._id;
    }
    set id(value: number){
        console.log(`set id property`);
        this._id = value;
    }
}
let classWithAccessors = new ClassWithAccessors();
classWithAccessors.id = 10;
console.log(`classWithAccessors.id = ${classWithAccessors.id}`);

// Static functions
class StaticFunction{
    static printTwo(){
        console.log(`2`);
    }
}
StaticFunction.printTwo();

// STatic properties
class StaticProperty{
    static count = 0;
    updateCount(){
        StaticProperty.count++;
    }
}
let firstInstance = new StaticProperty();
let secondInstance = new StaticProperty();
firstInstance.updateCount();
console.log(`StaticProperty.count = ${StaticProperty.count}`);
secondInstance.updateCount();
console.log(`StaticProperty.count = ${StaticProperty.count}`);

// Namespaces
namespace FirstNameSpace {
    export class NameSpaceClass{}
    class NotExported{}
}
let nameSpaceClass = new FirstNameSpace.NameSpaceClass();
//let notExported = new FirstNameSpace.NotExported();

// INHERITANCE

// Interface Inheritance
interface IBase {
    id: number;
}

interface IDerivedFromBase extends IBase {
    name: string;
}

class IdNameClass implements IDerivedFromBase {
    id: number = 0;
    name: string = "nameString";
}

interface IBaseStringOrNumber {
    id: string | number;
}
interface IDerivedFromBaseNumber extends IBaseStringOrNumber {
    id: number;
}

interface IMultiple extends IDerivedFromBase, IDerivedFromBaseNumber {
    description: string;
}

let multipleObject: IMultiple = {
    id: 1,
    name: "myName",
    description: "myDescription"
};

// Class inheritance
class BaseClass implements IBase{
    id: number = 0;
}
class DerivedFromBaseClass 
    extends BaseClass
    implements IDerivedFromBase
{
    name: string = "nameString";
}

interface IFirstInterface{
    id: number;
}
interface ISecondInterface{
    name: string;
}
class MultipleInterfaces implements
    IFirstInterface,
    ISecondInterface
{
    id: number = 0;
    name: string = "nameString";
}


// The `super` function
class BaseClassWithCtor{
    private id: number;
    constructor(id: number){
        this.id = id;
    }
}
class DerivedClassWithCtor extends BaseClassWithCtor{
    private name: string;
    constructor(id: number, name: string){
        super(id);
        this.name = name;
    }
}

// Function Overriding
class BaseClassWithFn{
    print(text: string){
        console.log(`BaseClassWithFn.print(): ${text}`);
    }
}
class DerivedClassFnOverride extends BaseClassWithFn{
    print(text: string){
        console.log(`DerivedClassFnOverride.print(): ${text}`);
    }
}
let derivedClassFnOverride = new DerivedClassFnOverride();
derivedClassFnOverride.print("test");
class DerivedClassFnCallthrough extends BaseClassWithFn{
    print(text: string){
        super.print(`from DerivedClassFnCallthrough : ${text}`);
    }
}
let derivedCallthrough = new DerivedClassFnCallthrough();
derivedCallthrough.print("text");

// Protected
class BaseClassProtected {
    protected id: number;
    private name: string = "";
    constructor(id: number){
        this.id = id;
    }
}

class AccessProtected extends BaseClassProtected {
    constructor(id: number){
        super(id);
        console.log(`base.id = ${this.id}`);
        // console.log(`base.name = ${this.name}`); --> private 
    }
}

// Abstract classes
abstract class EmployeeBase {
    public id: number;
    public name: string;
    constructor(id: number, name: string){
        this.id = id;
        this.name = name;
    }
}
class OfficeWorker extends EmployeeBase{

}
class OfficeManager extends OfficeWorker{
    public employees: OfficeWorker[] = [];
}