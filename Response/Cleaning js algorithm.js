const crawl_result = {
        "1": [],
        "2": [],
        "3": [
            "Kripik bawang nya enak , sayang nya hancur aja, mungkin perlu dipikirkan kemasan dan cara pengirimannya"
        ],
        "4": [],
        "5": [
            "enak. renyah.",
            "",
            "Trmksh, paket sdah diterima dg baik.",
            "⭐️⭐️⭐️⭐️⭐️",
            "mantap pokonya mah... ",
            "gak cukup 2 plastik.. pingin lagi n lagi",
            "Renyah dan tidak berminyak.",
            "Enak, gurih..  pokoknya manteeb bngt deh, mks sis",
            ""
        ]
    }

const maxlen = 20;
const vocab_size = 5000;
const padding = 'post';
const truncating = 'post';

//concat all arrar list
let arr = []
arr = arr.concat(crawl_result['1']);
arr = arr.concat(crawl_result['2']);
arr = arr.concat(crawl_result['3']);
arr = arr.concat(crawl_result['4']);
arr = arr.concat(crawl_result['5']);

//console.log(arr)
//remove character except string and space, tolowercase and trim the string
arr = arr.map(function(item) {
  item_ = item.replace(/[^a-zA-Z ]/g, " "); 
  item_ = item_.replace(/  +/g, ' '); 
  item_ = item_.toLowerCase();  
  return item_.trim(); 
});
//console.log(arr)

//remove empity
arr = arr.filter(function(e){ return e === 0 || e });
//remove duplicate
arr = [...new Set(arr)];
console.log(arr)

let result = {
	'negative' : [],
  'neutral' : [],
  'positive' : []
}
for (let i = 0; i < arr.length; i++) {
  let score = predict(inputText);
	console.log(score)
}


function predict(inputText){

    const sequence = inputText.map(word => {
        let indexed = word2index[word];

        if (indexed === undefined){
            return 1; //change to oov value
        }
        return indexed;
    });

    const paddedSequence = padSequence([sequence], maxlen);

    const score = tf.tidy(() => {
        const input = tf.tensor2d(paddedSequence, [1, maxlen]);
        console.log(input);
        const result = model.predict(input);
        return result.dataSync()[0];
    });

    return score;

}

function padSequence(sequences, maxLen, padding='post', truncating = "post", pad_value = 0){
    return sequences.map(seq => {
        if (seq.length > maxLen) { //truncat
            if (truncating === 'pre'){
                seq.splice(0, seq.length - maxLen);
            } else {
                seq.splice(maxLen, seq.length - maxLen);
            }
        }

        if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; i++){
                pad.push(pad_value);
            }
            if (padding === 'pre') {
                seq = pad.concat(seq);
            } else {
                seq = seq.concat(pad);
            }
        }
        return seq;
        });
}


