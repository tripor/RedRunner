using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace RedRunner.TerrainGeneration
{

    public abstract class Block : MonoBehaviour
    {

        [SerializeField]
        protected float m_Width;
        [SerializeField]
        protected float m_Probability = 1f;
        [SerializeField]
        protected float m_Restart_X = 2f;
        [SerializeField]
        protected float m_Restart_Y = 4f;
        [SerializeField]
        protected int identifier = 0;

        public virtual float Width
        {
            get
            {
                return m_Width;
            }
            set
            {
                m_Width = value;
            }
        }

        public virtual float Probability
        {
            get
            {
                return m_Probability;
            }
        }

        public virtual float Restart_X
        {
            get
            {
                return m_Restart_X;
            }
        }

        public virtual float Restart_Y
        {
            get
            {
                return m_Restart_Y;
            }
        }

        public virtual int Identifier
        {
            get
            {
                return this.identifier;
            }
        }

        public virtual void OnRemove(TerrainGenerator generator)
        {

        }

        public virtual void PreGenerate(TerrainGenerator generator)
        {

        }

        public virtual void PostGenerate(TerrainGenerator generator)
        {

        }

    }

}