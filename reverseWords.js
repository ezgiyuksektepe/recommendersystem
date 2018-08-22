function reverseString(str){
  if( typeof str !== "string"){
     throw "invalid input";
  }
  var newS = "";
  for(var i = str.length-1; i>=0; i--){
    newS+=str[i];
  }
  return newS;
}
function reverseWords(str){
   if( typeof str !== "string"){
     throw "invalid input";
   }
   var words = str.split(" ");
   var res = words.map(reverseString);
   return res.join(" ");
} 
