using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace RedRunner.UI
{
    public class UIHeartCounter : MonoBehaviour
    {
        public UIHeartsImage heart1;
        public UIHeartsImage heart2;
        public UIHeartsImage heart3;
        private int lives = 3;
        protected void Start()
        {
            RedRunner.Characters.RedCharacter.OnHeartLoss += RedCharacter_OnHeartLoss;
            RedRunner.Characters.RedCharacter.OnHeartReset += RedCharacter_OnHeartReset;
            if (GameManager.Singleton.simple_game)
            {
                lives = 1;
                heart2.EmptyHeart();
                heart3.EmptyHeart();
            }
            else
            {
                lives = 3;
            }
        }

        void RedCharacter_OnHeartLoss()
        {
            switch (lives)
            {
                case 3:
                    heart3.EmptyHeart();
                    break;
                case 2:
                    heart2.EmptyHeart();
                    break;
                case 1:
                    heart1.EmptyHeart();
                    break;
                default:
                    break;
            }
            lives--;
        }

        void RedCharacter_OnHeartReset()
        {
            if (GameManager.Singleton.simple_game)
            {
                lives = 2;
                heart1.ResetHeart();
            }
            else
            {
                lives = 3;
                heart3.ResetHeart();
                heart2.ResetHeart();
                heart1.ResetHeart();
            }
        }
    }
}