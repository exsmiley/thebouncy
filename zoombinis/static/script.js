let myId;
let myZ = [];
let zoombiniStrs = []
let gameOver = false;
let myCurrentZ = null;
let won = false;
let isClicking = false;

$.get('/api/start', function(data) {
    myId = data.id;

    for(let zoombini of data.zoombinis) {
        const zoombiniStr = `<div style="position: relative; height: 68px; width: 70px" id="z${zoombini.id}" onClick="handleClickZ(this)">
          <img src="static/zoo2/f${zoombini.feet}.png" style="position:absolute; zoom: 50%;"/>
          <img src="static/zoo2/e${zoombini.eyes}.png" style="position:absolute; zoom: 50%;"/>
          <img src="static/zoo2/h${zoombini.hair}.png" style="position:absolute; zoom: 50%;"/>
          <img src="static/zoo2/n${zoombini.nose}.png" style="position:absolute; zoom: 50%;"/>
        </div>`;
        myZ.push(zoombini);
        zoombiniStrs.push(zoombiniStr)
        $('#yours').append(zoombiniStr);
    }

});

function handleClickZ(element) {
    if(isClicking) {
        $('#buttons').html('')
        $('#z'+myCurrentZ.id).css('border', 'none')
    }
    if(!gameOver) {
        isClicking = true;
        let index = element.id;
        myCurrentZ = myZ[index.slice(1)];
        if(!myCurrentZ.has_passed) {
            $('#z'+myCurrentZ.id).css('border', 'dashed')
            $('#buttons').append('<button type="button" class="btn btn-primary" id="top" onClick="send(this)">Top</button><button type="button" class="btn btn-danger" id="bottom" onClick="send(this)">Bottom</button>');
        }
    }
}

function send(element) {
    isClicking = false;
    if(!gameOver) {
        let bridge = 0;
        if(element.id === "bottom") {
            bridge = 1;
        }

        let data = {'zoombini': myCurrentZ.id, 'bridge': bridge, 'id': myId};

        $.ajax({
          url:'/api/send',
          type: "POST",
          data: JSON.stringify(data),
          contentType: "application/json", // this
          dataType: "json", // and this
          success: function(result){
            console.log(result)
            $('#buttons').html('')
            gameOver = result.game_over;
            won = result.won;
            let passed = result.passed;
            myCurrentZ.has_passed = passed;

            $('#z' + myCurrentZ.id).remove()

            if(passed) {
                // top
                if(!bridge) {
                    $('#passedTop').append(zoombiniStrs[myCurrentZ.id])
                }
                // bottom
                else {
                    $('#passedBottom').append(zoombiniStrs[myCurrentZ.id])
                }
            } else {
                // top
                if(!bridge) {
                    $('#failedTop').append(zoombiniStrs[myCurrentZ.id])
                }
                // bottom
                else {
                    $('#failedBottom').append(zoombiniStrs[myCurrentZ.id])
                }
            }

            if(gameOver) {
                $('#buttons').append('<h2>Game Over</h2>')
                if(won) {
                    $('#buttons').append('<br><h2>You won!</h2>')
                } else {
                    $('#buttons').append('<br><h2>You won!</h2>')
                }
            }
          }
      })
    }
}



